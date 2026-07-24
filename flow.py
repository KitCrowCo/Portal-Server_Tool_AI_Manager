""" Generalized DAG Architecture (FlowCanvas Core)

## 1. Project Overview
This project is a generalized node-based graph architecture (Directed Acyclic Graph) designed to construct, modify, view, and interact with complex structural workflows, such as AI generation pipelines.
It operates as an independent, highly extensible data framework built to interface seamlessly with modern, unitless (SVG-style) canvas UIs while retaining the ability to tie into strict units when required.

It heavily emphasizes **Memory-First Graph Operations**. The in-memory data classes operate entirely independently of the presentation layer.
This ensures that structural linking and data payload mapping remain perfectly intact, strictly isolating topological integrity from visual "save states" or coordinate systems.

## 2. Core Architecture
The system is divided into two primary halves: the **Data Layer** (the source of truth) and the **Presentation Layer** (the UI canvas implementation).

### The Data Layer
*   **`FlowNode`:** A discrete node representing a specific entity, task, or data payload. It inherits standard dict-like behaviors for storing flexible data (e.g., `node["prompt"]`, `node["model"]`), while strictly isolating topological tracking (`prev`, `next`), identifiers, and nested UI `appearance` variables.
*   **`Flow`:** The memory-first DAG manager holding a dictionary of `FlowNode` objects. It is fully responsible for all topological edits—managing subflows, seamlessly splicing new node groupings between existing parent/child IDs (`insert_between`), and ensuring connection sanity on node deletion (`pop`) or path severance (`break_path`).
*   **Utility Infrastructure:** A suite of generalized tools built for high-performance pipeline support: fast custom JSON serialization/deserialization for non-standard types (`JsonManager`), deep-dictionary parsing without extensive recursive looping (`DictTools`), and temporal metadata approximations (`TimeWindow`).

### The Presentation Layer (Agnostic UI Implementation)
*   **Canvas Container:** The main GUI wrapper housing toolbars, node generation menus, and action bars (CRUD operations).
*   **Infinite Canvas:** A scalable 2D view (utilizing unitless/relative coordinates similar to SVG) that coordinates visual items, interactions, panning, and zooming. It reads from the `Flow` object to build the graph.
*   **Node Items:** The visual representation of a `FlowNode`. Listens for positional updates, styling overrides, and user interactions, feeding `(x,y)` coordinates back into the `node.appearance["pos"]` dictionary.
*   **Path Items:** The connective directional splines drawn dynamically between source and destination Node Items.

## 3. Data Integrity & Operation Paradigm
1.  **Strict State Management:** No structural edits (connecting nodes, deleting nodes) happen solely in the UI.
        UI interactions pass requests to the `Flow` object (e.g., `flow.break_path`, `flow.insert_between`).
        Once the data layer confirms the operation is topologically sound, the canvas rebuilds or updates the visual items.
2.  **Generalized Node Subsets:** Subsets of nodes can be temporarily tracked outside the main flow to allow visually isolated configurations or parameter tuning before permanently merging them into the active execution dataset.
3.  **Cascading Appearance Overrides:** Node visual styles cascade from bottom to top priorities: Canvas visual defaults -> `Flow`-level `appearance` dictates -> individual `FlowNode` `appearance` overrides.

---

flow.py — Segmented DAG core for ai_manager pipelines.

FlowNode/Flow are the source of truth for pipeline structure.
A pipeline IS a Flow: each FlowNode's payload is {"type": step_type, "config": {...}}; edges (prev/next) are execution order, branches are concurrent, merges wait on incoming prev (all or select).
Subflows: a node's payload may be {"type": "subflow", "flow": <nested Flow dict>} - engine.py treats this as one step that recursively runs the nested Flow and folds its terminal node outputs back as this node's result.
Arbitrarily deep nesting works because Flow.to_dict()/FlowNode are the same shape at every level.
"""

import os, json, math, uuid
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

class DictTools:
    """Generalized deep-dictionary manipulation and parsing tools."""

    @staticmethod
    def get(node: dict, path: list) -> Any:
        for key in path: node = node.get(key, {})
        return node

    @staticmethod
    def set(node: dict, path: list, value: Any):
        for key in path[:-1]: node = node.setdefault(key, {})
        node[path[-1]] = value

    @staticmethod
    def move(node: dict, old_path: list, new_path: list):
        DictTools.set(node, new_path, DictTools.get(node, old_path))
        DictTools.delete(node, old_path)

    @staticmethod
    def delete(node: dict, path: list):
        for key in path[:-1]: node = node.get(key, {})
        node.pop(path[-1], None)

    @staticmethod
    def clean(data: Any) -> Any:
        """Recursively removes empty lists and dictionaries in a single highly-efficient pass."""
        if isinstance(data, dict): return {k: v_clean for k, v in data.items() if (v_clean := DictTools.clean(v))}
        if isinstance(data, list): return [v_clean for v in data if (v_clean := DictTools.clean(v))]
        return data

    @staticmethod
    def compact_parse(data: Any, continue_func: Callable = None, format_func: Callable = None) -> Any:
        """A recursive dict parser. Instead of passing massive kwargs, it takes modular lambda/functions."""
        if isinstance(data, dict): 
            # Format current level, then recursively parse children if they meet the continue criteria
            parsed = {k: format_func(v) if format_func else v for k, v in data.items()}
            for k, v in data.items():
                if isinstance(v, (dict, list)) and (not continue_func or continue_func(k, v)): parsed[k] = DictTools.compact_parse(v, continue_func, format_func)
            return parsed
        if isinstance(data, list): return [DictTools.compact_parse(v, continue_func, format_func) for v in data]
        return data

class JsonManager:
    """Flexible JSON serialization toolbox supporting non-standard types (sets, tuples). Uses a targeted token prefix to identify encoded types on deserialization."""
    def __init__(self, token_prefix: str = "*^direct:"): self.prefix = token_prefix

    def encode(self, obj: Any) -> Any:
        """Recursively wraps unsupported Python types into tokenized strings."""
        if isinstance(obj, set): return f"{self.prefix}set:" + json.dumps(list(obj))
        if isinstance(obj, tuple): return f"{self.prefix}tuple:" + json.dumps(list(obj))
        if isinstance(obj, dict): return {str(k): self.encode(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self.encode(v) for v in obj]
        return obj

    def decode(self, obj: Any) -> Any:
        """Recursively unwraps tokenized strings back into native Python types."""
        if isinstance(obj, str) and obj.startswith(self.prefix):
            type_str, payload = obj[len(self.prefix):].split(":", 1)
            try:
                if type_str == "set": return set(json.loads(payload))
                if type_str == "tuple": return tuple(json.loads(payload))
            except json.JSONDecodeError:
                pass # Fallback to returning the raw string if parsing fails
        if isinstance(obj, dict): return {self.decode(k): self.decode(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self.decode(v) for v in obj]
        return obj

    def dump(self, data: Any, filepath: str, **kwargs):
        with open(filepath, 'w', encoding='utf-8') as f: json.dump(self.encode(data), f, **kwargs)

    def load(self, filepath: str) -> Any:
        with open(filepath, 'r', encoding='utf-8') as f: return self.decode(json.load(f))

    def dumps(self, data: Any, **kwargs) -> str: return json.dumps(self.encode(data), **kwargs)
    def loads(self, json_str: str) -> Any: return self.decode(json.loads(json_str))

@dataclass
class TimeWindow:
    """Truncated Gaussian representation for temporal metadata."""
    t_start: float
    t_end: float
    mu: Optional[float] = None
    s_start: float = 60.0
    s_end: float = 60.0
    k: float = 2.0
    unit: str = "s"

    def __post_init__(self):
        self.mu = self.mu if self.mu is not None else 0.5 * (self.t_start + self.t_end)
        if self.t_end <= self.t_start: raise ValueError("t_end must exceed t_start")
        self.sigma = (self.t_end - self.t_start) / (2.0 * max(1e-12, self.k))

    def sample(self, resolution: float = None) -> Tuple[np.ndarray, np.ndarray]:
        res = resolution or min(max(1e-6, self.s_start), max(1e-6, self.s_end))
        t = np.arange(self.t_start, self.t_end + 1e-9, res)

        # Calculate raw PDF and truncate
        raw_pdf = np.exp(-0.5 * ((t - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2.0 * np.pi))
        mask = (t >= self.t_start) & (t <= self.t_end)
        raw_pdf *= mask
        integral = np.trapz(raw_pdf[mask], t[mask])
        return t, (np.zeros_like(t) if integral <= 0 else raw_pdf / integral)

    def stats(self) -> Dict[str, float]:
        t, pdf = self.sample()
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1] if cdf.size > 0 else np.array([])
        return {"mean": float(np.trapz(t * pdf, t)) if pdf.sum() > 0 else self.mu, "p25": float(t[min(max(0, np.searchsorted(cdf, 0.25)), len(t)-1)]) if cdf.size > 0 else self.mu, "p50": float(t[min(max(0, np.searchsorted(cdf, 0.50)), len(t)-1)]) if cdf.size > 0 else self.mu, "p75": float(t[min(max(0, np.searchsorted(cdf, 0.75)), len(t)-1)]) if cdf.size > 0 else self.mu}

class TimeTranslator:
    """Handles time unit conversions and human-readable formatting."""
    FACTORS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}

    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> float: return (value * cls.FACTORS[from_unit]) / cls.FACTORS[to_unit]

    @classmethod
    def auto_label(cls, seconds: float) -> str:
        if seconds < 120: return f"{seconds:.1f} sec"
        if seconds < 7200: return f"{seconds/60:.1f} min"
        if seconds < 172800: return f"{seconds/3600:.1f} hr"
        return f"{seconds/86400:.1f} d"

    @staticmethod
    def format_window(tw: TimeWindow) -> Dict[str, str]:
        """Translates a TimeWindow into simplified UI approximations."""
        stats = tw.stats()
        fmt = lambda x: f"{x:g} {tw.unit}".strip()
        return {"window": f"{fmt(tw.t_start)} → {fmt(tw.t_end)}", "peak": fmt(tw.mu), "mean": fmt(stats["mean"]), "p25": fmt(stats["p25"]), "p50": fmt(stats["p50"]), "p75": fmt(stats["p75"])}

class FlowNode:
    """A generalized DAG node that acts as a dict for data payload while managing structural properties natively."""
    def __init__(self, data: dict = None):
        data = data or {}
        self.id: str = data.get("id", str(uuid.uuid4()))
        self.id2: str = data.get("id2", "")
        self.prev: List[str] = data.get("prev", [])
        self.next: List[str] = data.get("next", [])
        self.appearance: dict = data.get("appearance", {"pos": (0, 0)}) # UI State
        self._payload: dict = {k: v for k, v in data.items() if k not in ["id", "id2", "prev", "next", "appearance"]} # Data Payload

    def get(self, key: str, default: Any = None) -> Any: return self._payload.get(key, default)
    def __getitem__(self, key: str) -> Any: return self._payload[key]
    def __setitem__(self, key: str, value: Any): self._payload[key] = value
    def to_dict(self) -> dict: return {"id": self.id, "id2": self.id2, "prev": self.prev, "next": self.next, "appearance": self.appearance, **self._payload} # Serializes node back to a flat dictionary.

class Flow:
    """Memory-first DAG manager for FlowNodes, supporting insertions, deletions, and hierarchical properties."""
    def __init__(self, flow_data: Union[List[dict], Dict[str, Any]] = None):
        self.nodes: Dict[str, FlowNode] = {}
        self.appearance: dict = {}
        self.subflows: dict = {}
        if not flow_data: return
        if isinstance(flow_data, list): # Init from a flat list of dicts (e.g., when generating new stub nodes via UI)
            for n_data in flow_data:
                node = FlowNode(n_data)
                self.nodes[node.id] = node
        elif isinstance(flow_data, dict): # Init from a saved state dict containing subflows and global appearances
            for n_data in flow_data.get("nodes", []): 
                node = FlowNode(n_data)
                self.nodes[node.id] = node
            self.appearance = flow_data.get("appearance", {})
            self.subflows = flow_data.get("subflows", {})

    def __getitem__(self, index: int) -> FlowNode: return list(self.nodes.values())[index] # Allows for index-based access: flow[0] (used in UI to grab the first generated stub node)

    def insert_between(self, sub_flow: 'Flow', p_node_id: Optional[str] = None, n_node_id: Optional[str] = None):
        """Splices an entire sub-flow (group of nodes) between a given parent and child node ID."""
        for node_id, node in sub_flow.nodes.items():  self.nodes[node_id] = node # Absorb nodes into main tracker
        sub_heads = [n for n in sub_flow.nodes.values() if not n.prev]
        sub_tails = [n for n in sub_flow.nodes.values() if not n.next]
        if p_node_id and p_node_id in self.nodes: 
            p_node = self.nodes[p_node_id] # Connect parent to heads of the sub_flow
            for head in sub_heads:
                if head.id not in p_node.next: p_node.next.append(head.id)
                if p_node_id not in head.prev: head.prev.append(p_node_id)
        if n_node_id and n_node_id in self.nodes: 
            n_node = self.nodes[n_node_id] # Connect tails of the sub_flow to child
            for tail in sub_tails:
                if n_node_id not in tail.next: tail.next.append(n_node_id)
                if tail.id not in n_node.prev: n_node.prev.append(tail.id)  
        if p_node_id and n_node_id: self.break_path(p_node_id, n_node_id) # Break previous direct connection between p_node and n_node if it exists

    def pop(self, node_id: str) -> Optional[FlowNode]:
        """Safely removes a node from the graph and cleans up orphaned connections natively."""
        if node_id not in self.nodes: return None
        node = self.nodes.pop(node_id)
        for p_id in node.prev: # Scrub node ID from parent connections
            if p_id in self.nodes and node_id in self.nodes[p_id].next: 
                self.nodes[p_id].next.remove(node_id)
        for n_id in node.next: # Scrub node ID from child connections
            if n_id in self.nodes and node_id in self.nodes[n_id].prev: self.nodes[n_id].prev.remove(node_id)
        return node

    def break_path(self, src_id: str, dst_id: str):
        """Severs a directional path link between two node IDs."""
        if src_id in self.nodes and dst_id in self.nodes[src_id].next: self.nodes[src_id].next.remove(dst_id)
        if dst_id in self.nodes and src_id in self.nodes[dst_id].prev: self.nodes[dst_id].prev.remove(src_id)

    def to_dict(self) -> dict: return {"nodes": [node.to_dict() for node in self.nodes.values()], "appearance": self.appearance, "subflows": self.subflows} # Serializes entire flow object for JSON saving.
    def heads(self) -> List[FlowNode]: return [n for n in self.nodes.values() if not n.prev]
    def is_complete(self, done: set, skipped: set = None) -> bool: return all(n.id in (done | (skipped or set())) for n in self.nodes.values())

    def ready(self, done: set, skipped: set = None) -> List[FlowNode]:
    """Nodes not yet resolved whose join condition against prev is satisfied.
    join='all' (default): every prev must be done or skipped for this node to be considered - the caller is responsible for deciding whether an all-join node with a skipped prev should itself run or cascade to skipped, since that's an execution-semantics call, not a structural one.
    join='any': at least one prev must be done (skipped prevs don't block and don't count)."""
    skipped = skipped or set()
    resolved = done | skipped
    out = []
    for n in self.nodes.values():
        if n.id in resolved: continue
        if not n.prev: out.append(n); continue
        join = n.get("join", "all")
        if join == "any":
            if any(p in done for p in n.prev): out.append(n)
        elif all(p in resolved for p in n.prev): out.append(n)
    return out