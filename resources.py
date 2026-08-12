"""resources.py - CNode registry and pool resolution for ai_manager pipelines.
A CNode is one physical/logical machine hosting one or more connections (ollama, lightrag, flux2_text, flux2_image, ...).
Pools are plain set-intersection/exclusion over CNode tags and ids, resolved fresh on every call - no caching,  or historical optimization (that's the sDAG layer, deliberately deferred).
Selection within a pool is a single static score per priority mode using each CNode's own declared compute_score/quality_score - rough numbers today, expected to be refined once real usage data (logged below) accumulates.
"""
import json, time
from pathlib import Path
from typing import Optional
from tools.ai_manager import connections

CNODE_DIR = Path("./data/ai_manager/cnodes")
USAGE_LOG = Path("./data/ai_manager/_resource_log.jsonl")
CNODE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CNODE = {"label": "", "tags": [], "conn_ids": [], "mem_gb": 16.0, "overhead_gb": 2.0, "compute_score": 1.0, "quality_score": 1.0, "notes": ""}

def _cp(cid: str) -> Path: return CNODE_DIR / f"{Path(cid).name}.json"

def list_cnodes() -> list:
    out = []
    for f in sorted(CNODE_DIR.glob("*.json")):
        try: c = json.loads(f.read_text()); c["id"] = f.stem; out.append(c)
        except Exception: continue
    return out

def get_cnode(cid: str) -> Optional[dict]:
    p = _cp(cid)
    if not p.exists(): return None
    c = json.loads(p.read_text()); c["id"] = cid
    return c

def save_cnode(cid: str, data: dict): _cp(cid).write_text(json.dumps(data, indent=2))
def delete_cnode(cid: str): _cp(cid).unlink(missing_ok=True)

# --- Pool Resolution ---
# Pure successive set-filtering: Network (all CNodes) -> Pipeline pool (whitelist/blacklist tags+ids) -> Node's own cnode_tags (further intersected). Each level can only narrow, never widen, the level above it.

def resolve_candidates(pool_cfg: dict, node_tags: list = None, conn_type: str = "") -> list:
    wl_tags, bl_tags = set(pool_cfg.get("whitelist_tags", [])), set(pool_cfg.get("blacklist_tags", []))
    wl_ids, bl_ids = set(pool_cfg.get("whitelist_cnodes", [])), set(pool_cfg.get("blacklist_cnodes", []))
    ntags = set(node_tags or [])
    out = []
    for c in list_cnodes():
        ctags = set(c.get("tags", []))
        if c["id"] in bl_ids or (bl_tags and ctags & bl_tags): continue
        if wl_ids and c["id"] not in wl_ids: continue
        if wl_tags and not (ctags & wl_tags): continue
        if ntags and not (ctags & ntags): continue
        if conn_type and not any((connections.load_conn_raw(cid) or {}).get("connection_type") == conn_type for cid in c.get("conn_ids", [])): continue
        out.append(c)
    return out

def pick_conn(candidates: list, conn_type: str, priority: str = "balanced") -> Optional[tuple]:
    """Returns (cnode, conn_dict_with_id) for the best-scoring candidate's first matching connection, or None if nothing qualifies.
    Callers should fail loudly on None - an empty pool after filtering is a configuration problem worth surfacing, not papering over."""
    scored = []
    for c in candidates:
        for cid in c.get("conn_ids", []):
            conn = connections.load_conn_raw(cid)
            if not conn or conn.get("connection_type") != conn_type: continue
            score = {"speed": c.get("compute_score", 1.0), "quality": c.get("quality_score", 1.0)}.get(priority, (c.get("compute_score", 1.0) + c.get("quality_score", 1.0)) / 2)
            scored.append((score, c, cid, conn))
    if not scored: return None
    scored.sort(key=lambda x: x[0], reverse=True)
    _, cnode, cid, conn = scored[0]
    conn = dict(conn); conn["_id"] = cid
    return cnode, conn

def log_usage(cnode_id: str, node_type: str, elapsed_s: float, extra: dict = None):
    """Appends one line per node execution that went through pool resolution - passive dataset for a future sDAG cost model. Nothing reads this yet; safe to delete the file at any time."""
    try:
        with USAGE_LOG.open("a") as f: f.write(json.dumps({"t": time.time(), "cnode_id": cnode_id, "node_type": node_type, "elapsed_s": round(elapsed_s, 3), **(extra or {})}) + "\n")
    except Exception: pass

def resolve_bound(mode: str, target, bound=None, fallback=None):
    """Resolves one scalar against a goal-space spec instead of a single hardcoded number - exact / at_least / no_more_than / range.
    Generic on purpose: batch size, resolution, step count, or anything else that later wants 'as small as possible within X' semantics reuses this same small vocabulary rather than each field inventing its own bounds handling.
    fallback is today's best-known current value (currently just the field's own default); once AI Calc / real memory estimation is wired in, fallback becomes the place that plugs in."""
    if target in (None, ""): return fallback
    target = float(target)
    if mode == "exact": return target
    if mode == "at_least": return max(target, fallback) if fallback is not None else target
    if mode == "no_more_than": return min(target, fallback) if fallback is not None else target
    if mode == "range":
        lo, hi = target, float(bound) if bound not in (None, "") else target
        return max(lo, min(hi, fallback)) if fallback is not None else (lo + hi) / 2
    return fallback if fallback is not None else target