"""engine.py - job execution over ai_manager Flow graphs.
Pure key-presence scheduling: a node runs once every actual pipeline key it needs is present in the job's shared `data` object; no prev/next edges are consulted anywhere in this file.
Every currently-ready node runs together in one concurrent wave.
The job reaches a fixed point (no more nodes ready) either because everything ran, or because some nodes never got their keys - those are left status="unreached", which is the entire mechanism for conditional branches (see steps.py's branch/pipeline node types): nothing here ever 'decides' to skip anything.
"""

import json, uuid, asyncio, traceback, time
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from tools.ai_manager.flow import Flow, FlowNode
from tools.ai_manager.steps import get_node_type, NodeContext

ENV: dict = {}
PIPE_DIR = Path("./data/ai_manager/pipelines")
JOB_DIR = Path("./data/ai_manager/jobs")
_ACTIVE: dict = {}
logger = logging.getLogger("ai_manager.engine")

DEFAULT_POOL = {"whitelist_tags": [], "blacklist_tags": [], "whitelist_cnodes": [], "blacklist_cnodes": [], "priority": "balanced"}

# --- Pipeline definitions ---

def _pdp(pid: str) -> Path: return PIPE_DIR / f"{Path(pid).name}.json"
def load_pipeline(pid: str) -> Optional[dict]: p = _pdp(pid); return json.loads(p.read_text()) if p.exists() else None
def save_pipeline(doc: dict): doc["modified"] = datetime.utcnow().isoformat(); _pdp(doc["id"]).write_text(json.dumps(doc, indent=2))
def list_pipelines(tag: str = None) -> list: return [p for p in (json.loads(f.read_text()) for f in sorted(PIPE_DIR.glob("*.json"))) if not tag or tag in p.get("tags", [])]
def new_pipeline(owner: str, name: str = "New Pipeline") -> dict: return {"id": f"pl_{uuid.uuid4().hex[:10]}", "name": name, "owner": owner, "tags": [], "flow": Flow().to_dict(), "pool": dict(DEFAULT_POOL), "created": datetime.utcnow().isoformat()}
def delete_pipeline(pid: str): _pdp(pid).unlink(missing_ok=True)
def delete_job(jid: str): _jdp(jid).unlink(missing_ok=True)

# --- Jobs ---

def _jdp(jid: str) -> Path: return JOB_DIR / f"{Path(jid).name}.json"
def load_job(jid: str) -> Optional[dict]: p = _jdp(jid); return json.loads(p.read_text()) if p.exists() else None
def _save_job(job: dict): job["modified"] = datetime.utcnow().isoformat(); _jdp(job["id"]).write_text(json.dumps(job, indent=2))

def init(env: dict):
    global ENV
    ENV = env
    for d in (PIPE_DIR, JOB_DIR): d.mkdir(parents=True, exist_ok=True)
    _reconcile_stale_jobs()

def _reconcile_stale_jobs():
    for f in JOB_DIR.glob("*.json"):
        try:
            job = json.loads(f.read_text())
            if job.get("status") == "running": job["status"] = "interrupted"; f.write_text(json.dumps(job, indent=2))
        except Exception: continue

def _log(job: dict, msg: str): job.setdefault("log", []).append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")

async def _set_node_status(job_id: str, username: str, node_id: str, status: str, extra: dict = None):
    """Known pre-existing race: two nodes finishing in the same concurrent wave can each load-modify-save this job doc and clobber one another's status write.
    Cosmetic only (data merge itself is unaffected, see _run_flow) - not fixed here, flagged for whoever eventually adds a lock around job file writes."""
    job = load_job(job_id)
    if not job: return
    node = next((n for n in job["flow"]["nodes"] if n["id"] == node_id), None)
    if node:
        node["status"] = status; node["ts"] = datetime.utcnow().strftime("%H:%M:%S")
        if extra: node.update(extra)
    job["heartbeat"] = time.time()
    _save_job(job)
    await ENV["push_to_client"](username, {"t": "pipeline_event", "job_id": job_id, "event": status, "payload": {"node": node_id, **(extra or {})}})

async def _run_flow(flow: dict, data: dict, job_id: str, username: str, pool_cfg: dict, depth: int = 0) -> dict:
    f = Flow(flow)
    done = {n.id for n in f.nodes.values() if n.status == "done"}
    wave_num = 0
    while True:
        job = load_job(job_id)
        if job["status"] == "stopping": _log(job, "stop requested - halting before next wave"); _save_job(job); break
        ready = []
        for n in f.nodes.values():
            if n.id in done: continue
            spec = get_node_type(n.type)
            if not spec: continue
            if f.required_in_keys(n, spec) <= set(data.keys()): ready.append((n, spec))
        if not ready: break
        wave_num += 1
        job = load_job(job_id); _log(job, f"wave {wave_num}: starting {len(ready)} node(s): {[n.id for n,_ in ready]}"); _save_job(job)

        async def run_one(n: FlowNode, spec: dict):
            await _set_node_status(job_id, username, n.id, "running", {"type": n.type, "name": n.name})
            t0 = time.time()
            try:
                ctx = NodeContext(n, data, job_id, username, pool_cfg, depth)
                result = await spec["fn"](n.config, data, ctx)
                for logical, val in (result or {}).items(): data[n.out_key(logical)] = val
                preview = {k: str(v)[:500] for k, v in (result or {}).items()}
                await _set_node_status(job_id, username, n.id, "done", {"preview": preview, "elapsed_s": round(time.time()-t0, 2)})
                return n.id
            except Exception as e:
                traceback.print_exc()
                await _set_node_status(job_id, username, n.id, "error", {"message": str(e)})
                raise

        try:
            finished = await asyncio.gather(*[run_one(n, spec) for n, spec in ready])
        except Exception as e:
            job = load_job(job_id); job["status"] = "error"; _log(job, f"wave {wave_num} failed: {e}"); _save_job(job)
            return data
        done.update(finished)
        job = load_job(job_id); job["data"] = data; _log(job, f"wave {wave_num}: complete"); _save_job(job)
    return data

async def _run(job_id: str):
    job = load_job(job_id)
    if not job: return
    job["status"] = "running"; _log(job, "job started"); _save_job(job)
    try:
        data = await _run_flow(job["flow"], dict(job["data"]), job_id, job["username"], job.get("pool", DEFAULT_POOL))
    except Exception as e:
        traceback.print_exc()
        job = load_job(job_id); job["status"] = "error"; _log(job, f"FATAL - job crashed outside normal node handling: {e}"); _save_job(job)
        _ACTIVE.pop(job_id, None)
        return
    job = load_job(job_id)
    flow_nodes = job["flow"]["nodes"]
    if job["status"] == "stopping": job["status"] = "stopped"
    elif any(n.get("status") == "error" for n in flow_nodes): job["status"] = "error"
    else:
        for n in flow_nodes:
            if n.get("status") not in ("done","error"): n["status"] = "unreached"
        job["status"] = "done"
    job["data"] = data
    _log(job, f"job finished with status: {job['status']}")
    _save_job(job)
    _ACTIVE.pop(job_id, None)

def _task_exception_logger(task: asyncio.Task):
    if task.cancelled(): return
    exc = task.exception()
    if exc: logger.error("Unhandled exception in pipeline job task", exc_info=exc)

def submit(username, kind="id", pipeline_id="", inline_flow=None, inputs=None, allowed_tags=None, allowed_ids=None, pool_cfg=None) -> tuple:
    if kind == "id":
        pdef = load_pipeline(pipeline_id)
        if not pdef: return None, "pipeline not found"
        if allowed_ids is not None and pipeline_id not in allowed_ids: return None, "pipeline not permitted for this caller"
        if allowed_tags is not None and not (set(pdef.get("tags", [])) & set(allowed_tags)): return None, "pipeline lacks a permitted tag"
        flow_data = json.loads(json.dumps(pdef["flow"]))
        pool = pool_cfg or pdef.get("pool", DEFAULT_POOL)
    else:
        flow_data = inline_flow
        if not flow_data: return None, "no inline flow provided"
        pool = pool_cfg or DEFAULT_POOL
    for n in flow_data.get("nodes", []): n["status"] = "idle"; n.pop("ts", None); n.pop("preview", None); n.pop("message", None)
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job = {"id": job_id, "username": username, "flow": flow_data, "pool": pool, "status": "queued", "data": dict(inputs or {}), "log": [], "heartbeat": time.time(), "created": datetime.utcnow().isoformat()}
    _save_job(job)
    task = asyncio.create_task(_run(job_id))
    task.add_done_callback(_task_exception_logger)
    _ACTIVE[job_id] = task
    return job_id, ""

def resume(job_id: str) -> tuple:
    job = load_job(job_id)
    if not job: return None, "job not found"
    if job["status"] == "running": return job_id, ""
    job["status"] = "running"; _save_job(job)
    task = asyncio.create_task(_run(job_id))
    task.add_done_callback(_task_exception_logger)
    _ACTIVE[job_id] = task
    return job_id, ""

def stop(job_id: str):
    job = load_job(job_id)
    if job: job["status"] = "stopping"; _save_job(job)

async def run_inline(username: str, pipeline_id: str, inputs: dict = None, pool_cfg: dict = None, depth: int = 0, max_depth = 100, job_id: str = None) -> dict:
    """Runs a saved pipeline to completion and returns its final data object - used by the pipeline/pipeline_foreach/branch node types to compose pipelines together. Depth guards against runaway self-referential recursion.
    job_id, if supplied by the caller, lets a parent node stash a stable reference to this sub-run's job record for later inspection (see steps.py's node_pipeline) - otherwise one is generated as before."""
    if depth > max_depth: raise RuntimeError(f"run_inline: max pipeline call depth ({max_depth}) exceeded - likely an unbounded recursive branch")
    pdef = load_pipeline(pipeline_id)
    if not pdef: raise RuntimeError(f"run_inline: pipeline not found: {pipeline_id}")
    flow_data = json.loads(json.dumps(pdef["flow"]))
    for n in flow_data.get("nodes", []): n["status"] = "idle"; n.pop("ts", None); n.pop("preview", None); n.pop("message", None)
    pool = pool_cfg or pdef.get("pool", DEFAULT_POOL)
    job_id = job_id or f"job_{uuid.uuid4().hex[:10]}"
    job = {"id": job_id, "username": username, "flow": flow_data, "pool": pool, "status": "queued", "data": dict(inputs or {}), "log": [], "heartbeat": time.time(), "created": datetime.utcnow().isoformat()}
    _save_job(job)
    data = await _run_flow(flow_data, dict(job["data"]), job_id, username, pool, depth=depth)
    job = load_job(job_id)
    job["data"] = data
    job["status"] = "error" if any(n.get("status") == "error" for n in flow_data["nodes"]) else "done"
    _save_job(job)
    return data