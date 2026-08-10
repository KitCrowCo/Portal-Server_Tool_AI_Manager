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

flow.py — Universal-node pipeline core for ai_manager.
FlowNode/Flow hold node definitions only; execution order is never stored here, it's derived purely from which actual pipeline keys a node needs versus which other nodes produce those keys (see resolve_levels, used only for the graphical builder view, and engine.py's scheduler, which recomputes readiness fresh every wave against the live shared data object).
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
        if isinstance(data, dict): return {k: v_clean for k, v in data.items() if (v_clean := DictTools.clean(v))}
        if isinstance(data, list): return [v_clean for v in data if (v_clean := DictTools.clean(v))]
        return data

    @staticmethod
    def compact_parse(data: Any, continue_func: Callable = None, format_func: Callable = None) -> Any:
        if isinstance(data, dict):
            parsed = {k: format_func(v) if format_func else v for k, v in data.items()}
            for k, v in data.items():
                if isinstance(v, (dict, list)) and (not continue_func or continue_func(k, v)): parsed[k] = DictTools.compact_parse(v, continue_func, format_func)
            return parsed
        if isinstance(data, list): return [DictTools.compact_parse(v, continue_func, format_func) for v in data]
        return data

class JsonManager:
    """Flexible JSON serialization toolbox supporting non-standard types (sets, tuples)."""
    def __init__(self, token_prefix: str = "*^direct:"): self.prefix = token_prefix

    def encode(self, obj: Any) -> Any:
        if isinstance(obj, set): return f"{self.prefix}set:" + json.dumps(list(obj))
        if isinstance(obj, tuple): return f"{self.prefix}tuple:" + json.dumps(list(obj))
        if isinstance(obj, dict): return {str(k): self.encode(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self.encode(v) for v in obj]
        return obj

    def decode(self, obj: Any) -> Any:
        if isinstance(obj, str) and obj.startswith(self.prefix):
            type_str, payload = obj[len(self.prefix):].split(":", 1)
            try:
                if type_str == "set": return set(json.loads(payload))
                if type_str == "tuple": return tuple(json.loads(payload))
            except json.JSONDecodeError:
                pass
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
        stats = tw.stats()
        fmt = lambda x: f"{x:g} {tw.unit}".strip()
        return {"window": f"{fmt(tw.t_start)} → {fmt(tw.t_end)}", "peak": fmt(tw.mu), "mean": fmt(stats["mean"]), "p25": fmt(stats["p25"]), "p50": fmt(stats["p50"]), "p75": fmt(stats["p75"])}

class FlowNode:
    """Universal pipeline node. Agnostic to any specific step's shape - only knows its own type, config, and the logical<->actual key mapping that lets it read/write the pipeline's shared data object.
    No prev/next: execution order is derived entirely from key presence (engine.py), never stored here. appearance is UI-only and never read by execution."""
    def __init__(self, data: dict = None):
        data = data or {}
        self.id: str = data.get("id", str(uuid.uuid4()))
        self.type: str = data.get("type", "")
        self.name: str = data.get("name", "")
        self.config: dict = data.get("config", {})
        self.key_map: dict = data.get("key_map", {})                    # logical name -> actual pipeline key (applies to in AND out)
        self.extra_in_keys: List[str] = data.get("extra_in_keys", [])   # actual pipeline keys needed beyond the type's declared logical ins
        self.extra_out_keys: List[str] = data.get("extra_out_keys", []) # actual pipeline keys promised beyond the type's declared logical outs
        self.appearance: dict = data.get("appearance", {})
        self.status: str = data.get("status", "idle")
        self.ts: str = data.get("ts", "")
        self.preview: dict = data.get("preview", {})
        self.message: str = data.get("message", "")

    def in_key(self, logical: str) -> str: return self.key_map.get(logical, logical)
    def out_key(self, logical: str) -> str: return self.key_map.get(logical, logical)
    def to_dict(self) -> dict: return {"id": self.id, "type": self.type, "name": self.name, "config": self.config, "key_map": self.key_map, "extra_in_keys": self.extra_in_keys, "extra_out_keys": self.extra_out_keys, "appearance": self.appearance, "status": self.status, "ts": self.ts, "preview": self.preview, "message": self.message}

class Flow:
    """Memory-first container for a set of universal FlowNodes. No structural edges are stored - a node's causal position is entirely a function of which actual pipeline keys it needs versus which other nodes produce those keys, recomputed on demand.
    Nodes can be freely added, removed, or rewired (by editing key_map) without a second data structure to keep in sync."""
    def __init__(self, flow_data: Union[List[dict], Dict[str, Any]] = None):
        self.nodes: Dict[str, FlowNode] = {}
        self.appearance: dict = {}
        if not flow_data: return
        if isinstance(flow_data, list):
            for n_data in flow_data: node = FlowNode(n_data); self.nodes[node.id] = node
        elif isinstance(flow_data, dict):
            for n_data in flow_data.get("nodes", []): node = FlowNode(n_data); self.nodes[node.id] = node
            self.appearance = flow_data.get("appearance", {})

    def __getitem__(self, index: int) -> FlowNode: return list(self.nodes.values())[index]
    def add(self, node: FlowNode): self.nodes[node.id] = node
    def remove(self, node_id: str) -> Optional[FlowNode]: return self.nodes.pop(node_id, None)
    def to_dict(self) -> dict: return {"nodes": [n.to_dict() for n in self.nodes.values()], "appearance": self.appearance}
    def required_in_keys(self, node: FlowNode, type_spec: dict) -> set: return {node.in_key(k) for k in type_spec.get("in_keys", [])} | set(node.extra_in_keys) # Actual pipeline keys that must be present in the shared data object before this node can run.
    def produced_out_keys(self, node: FlowNode, type_spec: dict) -> set: return {node.out_key(k) for k in type_spec.get("out_keys", [])} | set(node.extra_out_keys) # Actual pipeline keys this node promises to write - used only for the causal-level display, never for scheduling (scheduling reacts to keys that actually appear at runtime, not to what a node merely claims it might produce).

    def resolve_levels(self, type_specs: dict) -> Dict[str, int]:
        """Display-only: assigns each node a causal level (0 = no dependencies among current nodes) for the graphical builder view - time flows down, same level means potentially concurrent.
        Never consulted by the engine."""
        producers: Dict[str, list] = {}
        for n in self.nodes.values():
            spec = type_specs.get(n.type, {})
            for k in self.produced_out_keys(n, spec): producers.setdefault(k, []).append(n.id)
        level: Dict[str, int] = {}
        def _lvl(n: FlowNode, seen: set) -> int:
            if n.id in level: return level[n.id]
            if n.id in seen: return 0
            seen = seen | {n.id}
            spec = type_specs.get(n.type, {})
            deps = set()
            for k in self.required_in_keys(n, spec): deps.update(producers.get(k, []))
            deps.discard(n.id)
            lvl = 1 + max((_lvl(self.nodes[d], seen) for d in deps if d in self.nodes), default=-1)
            level[n.id] = lvl
            return lvl
        for n in self.nodes.values(): _lvl(n, set())
        return level