"""engine.py — job execution over ai_manager Flow graphs.
Ownership: this module holds every live connection for the duration of a job.
Callers never await network I/O directly - submit() returns a job_id immediately; all progress arrives over WS via push_to_client, addressed by job_id, so any module's UI can subscribe by matching that id in its own OOB targets.
Capability gating: pipelines carry tags (list[str]).
A caller supplies allowed_tags or allowed_ids when submitting on behalf of a restricted surface (e.g. Athena) - the engine refuses anything outside that allowlist.
Ad-hoc inline flows (kind="inline", no saved pipeline_id) are for Tessa's test-run surface and anywhere building a one-off flow at request time - callers choosing kind="inline" are responsible for not exposing that path to untrusted input.
"""
import json, uuid, asyncio, re, traceback, time
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from tools.ai_manager.flow import Flow
from tools.ai_manager.steps import get_step_type

ENV: dict = {}
PIPE_DIR = Path("./data/ai_manager/pipelines")
JOB_DIR = Path("./data/ai_manager/jobs")
_ACTIVE: dict = {}
logger = logging.getLogger("ai_manager.engine")

# -- Pipeline definitions --

def _pdp(pid: str) -> Path: return PIPE_DIR / f"{Path(pid).name}.json"
def load_pipeline(pid: str) -> Optional[dict]: p = _pdp(pid); return json.loads(p.read_text()) if p.exists() else None
def save_pipeline(doc: dict): doc["modified"] = datetime.utcnow().isoformat(); _pdp(doc["id"]).write_text(json.dumps(doc, indent=2))
def list_pipelines(tag: str = None) -> list: return [p for p in (json.loads(f.read_text()) for f in sorted(PIPE_DIR.glob("*.json"))) if not tag or tag in p.get("tags", [])]
def new_pipeline(owner: str, name: str = "New Pipeline") -> dict: return {"id": f"pl_{uuid.uuid4().hex[:10]}", "name": name, "owner": owner, "tags": [], "flow": Flow().to_dict(), "created": datetime.utcnow().isoformat()}
def delete_pipeline(pid: str): _pdp(pid).unlink(missing_ok=True)

# -- Jobs --

def _jdp(jid: str) -> Path: return JOB_DIR / f"{Path(jid).name}.json"
def load_job(jid: str) -> Optional[dict]: p = _jdp(jid); return json.loads(p.read_text()) if p.exists() else None
def _save_job(job: dict): job["modified"] = datetime.utcnow().isoformat(); _jdp(job["id"]).write_text(json.dumps(job, indent=2))    
    
STALE_SECONDS = 45  # no heartbeat in this window while status=="running" -> treat as interrupted, not silently stuck

def init(env: dict):
    global ENV
    ENV = env
    for d in (PIPE_DIR, JOB_DIR): d.mkdir(parents=True, exist_ok=True)
    _reconcile_stale_jobs()

def _reconcile_stale_jobs():
    """On startup, any job left mid-'running' has no live asyncio task behind it anymore (process restarted) - mark it interrupted so status is honest instead of stuck forever."""
    for f in JOB_DIR.glob("*.json"):
        try:
            job = json.loads(f.read_text())
            if job.get("status") == "running":
                job["status"] = "interrupted"
                f.write_text(json.dumps(job, indent=2))
        except Exception: continue

class StepContext:
    def __init__(self, job_id, username, scratch): self.job_id, self.username, self.scratch = job_id, username, scratch
    def resolve(self, template) -> str:
        """Must never return or propagate None - a template that is missing, None, or not a string always resolves to an empty string rather than crashing something two calls away."""
        if not isinstance(template, str): template = "" if template is None else str(template)
        def sub(m):
            path = m.group(1).split(".")
            val = self.scratch
            for p in path: val = val.get(p, "") if isinstance(val, dict) else ""
            return str(val) if val else ""
        return re.sub(r"\{([\w\.]+)\}", sub, template)
    async def set_node_status(self, node_id: str, status: str, extra: dict = None):
        job = load_job(self.job_id)
        if not job: return
        node = next((n for n in job["flow"]["nodes"] if n["id"] == node_id), None)
        if node:
            node["status"] = status
            node["ts"] = datetime.utcnow().strftime("%H:%M:%S")
            if extra: node.update(extra)
        job["heartbeat"] = time.time()
        _save_job(job)
        await ENV["push_to_client"](self.username, {"t": "pipeline_event", "job_id": self.job_id, "event": status, "payload": {"node": node_id, **(extra or {})}})
    async def stream(self, key, delta): await ENV["push_to_client"](self.username, {"t": "pipeline_stream", "job_id": self.job_id, "key": key, "delta": delta})

def _done_set(flow: dict) -> set: return {n["id"] for n in flow["nodes"] if n.get("status") == "done"}

def submit(username, kind="id", pipeline_id="", inline_flow=None, inputs=None, allowed_tags=None, allowed_ids=None, extra_config=None) -> tuple:
    if kind == "id":
        pdef = load_pipeline(pipeline_id)
        if not pdef: return None, "pipeline not found"
        if allowed_ids is not None and pipeline_id not in allowed_ids: return None, "pipeline not permitted for this caller"
        if allowed_tags is not None and not (set(pdef.get("tags", [])) & set(allowed_tags)): return None, "pipeline lacks a permitted tag"
        flow_data = json.loads(json.dumps(pdef["flow"]))  # deep copy - a job's flow is its own progress log, must not alias the saved definition
    else:
        flow_data = inline_flow
        if not flow_data: return None, "no inline flow provided"
    for n in flow_data.get("nodes", []): n["status"] = "idle"; n.pop("ts", None); n.pop("preview", None); n.pop("message", None)
    if extra_config:
        for n in flow_data.get("nodes", []): n.setdefault("config", {}).update(extra_config)

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job = {"id": job_id, "username": username, "flow": flow_data, "status": "queued", "scratch": {"input": (inputs or {}).get("input", "")}, "log": [], "heartbeat": time.time(), "created": datetime.utcnow().isoformat()}
    _save_job(job)
    task = asyncio.create_task(_run(job_id))
    task.add_done_callback(_task_exception_logger)
    _ACTIVE[job_id] = task
    return job_id, ""

def resume(job_id: str) -> tuple:
    """Re-enters an interrupted/stopped job, resuming from whichever nodes are not yet status=='done' in its saved flow."""
    job = load_job(job_id)
    if not job: return None, "job not found"
    if job["status"] == "running": return job_id, ""
    job["status"] = "running"
    _save_job(job)
    task = asyncio.create_task(_run(job_id))
    task.add_done_callback(_task_exception_logger)
    _ACTIVE[job_id] = task
    return job_id, ""

def stop(job_id: str):
    job = load_job(job_id)
    if job: job["status"] = "stopping"; _save_job(job)

def _log(job: dict, msg: str): job.setdefault("log", []).append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")

async def _run_flow(flow: dict, ctx: "StepContext", job_id: str) -> set:
    f = Flow(flow)
    done = _done_set(flow)
    wave_num = 0
    while not f.is_complete(done):
        job = load_job(job_id)
        if job["status"] == "stopping":
            _log(job, "stop requested - halting before next wave"); _save_job(job)
            return done
        wave = [n for n in f.ready(done) if n.to_dict().get("status") != "done"]
        if not wave:
            _log(job, f"no ready nodes and flow incomplete - dangling reference or cycle, stopping"); _save_job(job)
            break
        wave_num += 1
        job = load_job(job_id)
        _log(job, f"wave {wave_num}: starting {len(wave)} node(s): {[n.to_dict()['id'] for n in wave]}")
        _save_job(job)

        async def run_one(node):
            nd = node.to_dict()
            await ctx.set_node_status(nd["id"], "running", {"type": nd.get("type",""), "name": nd.get("name","")})
            try:
                if nd.get("type") == "subflow":
                    await _run_flow(nd.get("flow", {}), ctx, job_id)
                    result = dict(ctx.scratch)
                else:
                    spec = get_step_type(nd.get("type", ""))
                    if not spec: raise RuntimeError(f"unknown step type: {nd.get('type')}")
                    result = await spec["fn"](nd.get("config", {}), ctx)
                ctx.scratch[nd["id"]] = result
                preview = {k: str(v)[:100] for k, v in (result or {}).items()}
                await ctx.set_node_status(nd["id"], "done", {"preview": preview})
            except Exception as e:
                traceback.print_exc()
                await ctx.set_node_status(nd["id"], "error", {"message": str(e)})
                raise
            return nd["id"]

        try:
            results = await asyncio.gather(*[run_one(n) for n in wave])
            done.update(results)
        except Exception as e:
            job = load_job(job_id)
            job["status"] = "error"; _log(job, f"wave {wave_num} failed: {e}"); _save_job(job)
            return done
        job = load_job(job_id)
        job["scratch"] = ctx.scratch
        _log(job, f"wave {wave_num}: complete")
        _save_job(job)
    return done

async def _run(job_id: str):
    job = load_job(job_id)
    if not job: return
    job["status"] = "running"; _log(job, "job started"); _save_job(job)
    ctx = StepContext(job_id, job["username"], job["scratch"])
    try:
        done = await _run_flow(job["flow"], ctx, job_id)
    except Exception as e:
        # last-resort catch: if anything above this point throws, the job record itself says so, instead of silently dying as an unretrieved task exception.
        traceback.print_exc()
        job = load_job(job_id)
        job["status"] = "error"
        _log(job, f"FATAL - job crashed outside normal step handling: {e}")
        _save_job(job)
        _ACTIVE.pop(job_id, None)
        return
    job = load_job(job_id)
    flow_nodes = job["flow"]["nodes"]
    if job["status"] == "stopping": job["status"] = "stopped"
    elif any(n.get("status") == "error" for n in flow_nodes): job["status"] = "error"
    elif all(n.get("status") == "done" for n in flow_nodes): job["status"] = "done"
    else: job["status"] = "stopped"
    job["scratch"] = ctx.scratch
    _log(job, f"job finished with status: {job['status']}")
    _save_job(job)
    _ACTIVE.pop(job_id, None)

def _task_exception_logger(task: asyncio.Task):
    """Second safety net: even if _run's own try/except somehow doesn't catch something (e.g. a crash before job could be loaded at all), this makes it print loudly to the server console instead of vanishing as 'Task exception was never retrieved'."""
    if task.cancelled(): return
    exc = task.exception()
    if exc: logger.error("Unhandled exception in pipeline job task", exc_info=exc)
    
async def set_node_status(self, node_id: str, status: str, extra: dict = None):
    job = load_job(self.job_id)
    if not job: return
    node = next((n for n in job["flow"]["nodes"] if n["id"] == node_id), None)
    if node:
        node["status"] = status
        node["ts"] = datetime.utcnow().strftime("%H:%M:%S")
        if extra: node.update(extra)
    label = node.get("name") or node_id if node else node_id
    detail = f" - {extra['message']}" if extra and extra.get("message") else ""
    job.setdefault("log", []).append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {label}: {status}{detail}")
    job["heartbeat"] = time.time()
    _save_job(job)
    await ENV["push_to_client"](self.username, {"t": "pipeline_event", "job_id": self.job_id, "event": status, "payload": {"node": node_id, **(extra or {})}})