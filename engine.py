"""engine.py — job execution over ai_manager Flow graphs.
Ownership: this module holds every live connection for the duration of a job.
Callers never await network I/O directly - submit() returns a job_id immediately; all progress arrives over WS via push_to_client, addressed by job_id, so any module's UI can subscribe by matching that id in its own OOB targets.
Capability gating: pipelines carry tags (list[str]).
A caller supplies allowed_tags or allowed_ids when submitting on behalf of a restricted surface (e.g. Athena) - the engine refuses anything outside that allowlist.
Ad-hoc inline flows (kind="inline", no saved pipeline_id) are for Tessa's test-run surface and anywhere building a one-off flow at request time - callers choosing kind="inline" are responsible for not exposing that path to untrusted input.
"""
import json, uuid, asyncio, re, traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

# import sys, os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flow import Flow
from steps import get_step_type
# from .flow import Flow
# from .steps import get_step_type

ENV: dict = {}
PIPE_DIR = Path("./data/ai_manager/pipelines")
JOB_DIR = Path("./data/ai_manager/jobs")
_ACTIVE: dict = {}

def init(env: dict):
    global ENV
    ENV = env
    for d in (PIPE_DIR, JOB_DIR): d.mkdir(parents=True, exist_ok=True)

# -- Pipeline definitions --

def _pdp(pid: str) -> Path: return PIPE_DIR / f"{Path(pid).name}.json"
def load_pipeline(pid: str) -> Optional[dict]: p = _pdp(pid); return json.loads(p.read_text()) if p.exists() else None
def save_pipeline(doc: dict): doc["modified"] = datetime.utcnow().isoformat(); _pdp(doc["id"]).write_text(json.dumps(doc, indent=2))
def list_pipelines(tag: str = None) -> list: return [p for p in (json.loads(f.read_text()) for f in sorted(PIPE_DIR.glob("*.json"))) if not tag or tag in p.get("tags", [])]
def new_pipeline(owner: str, name: str = "New Pipeline") -> dict: return {"id": f"pl_{uuid.uuid4().hex[:10]}", "name": name, "owner": owner, "tags": [], "flow": Flow().to_dict(), "created": datetime.utcnow().isoformat()}

# -- Jobs --

def _jdp(jid: str) -> Path: return JOB_DIR / f"{Path(jid).name}.json"
def load_job(jid: str) -> Optional[dict]: p = _jdp(jid); return json.loads(p.read_text()) if p.exists() else None
def _save_job(job: dict): job["modified"] = datetime.utcnow().isoformat(); _jdp(job["id"]).write_text(json.dumps(job, indent=2))

class StepContext:
    def __init__(self, job_id: str, username: str, scratch: dict):
        self.job_id, self.username, self.scratch = job_id, username, scratch
    def resolve(self, template: str) -> str:
        """{input} -> scratch['input']; {node_id.key} -> scratch nested under that node's result dict."""
        def sub(m):
            path = m.group(1).split(".")
            val = self.scratch
            for p in path: val = val.get(p, "") if isinstance(val, dict) else ""
            return str(val) if val else ""
        return re.sub(r"\{([\w\.]+)\}", sub, template or "")
    async def push(self, event: str, payload: dict): await ENV["push_to_client"](self.username, {"t": "pipeline_event", "job_id": self.job_id, "event": event, "payload": payload})
    async def stream(self, key: str, delta: str): await ENV["push_to_client"](self.username, {"t": "pipeline_stream", "job_id": self.job_id, "key": key, "delta": delta})

def submit(username: str, kind: str = "id", pipeline_id: str = "", inline_flow: dict = None, inputs: dict = None, allowed_tags: list = None, allowed_ids: list = None) -> tuple:
    if kind == "id":
        pdef = load_pipeline(pipeline_id)
        if not pdef: return None, "pipeline not found"
        if allowed_ids is not None and pipeline_id not in allowed_ids: return None, "pipeline not permitted for this caller"
        if allowed_tags is not None and not (set(pdef.get("tags", [])) & set(allowed_tags)): return None, "pipeline lacks a permitted tag"
        flow_data = pdef["flow"]
    else:
        flow_data = inline_flow
        if not flow_data: return None, "no inline flow provided"

    job_id = f"job_{uuid.uuid4().hex[:10]}"
    job = {"id": job_id, "username": username, "flow": flow_data, "status": "queued", "scratch": {"input": (inputs or {}).get("input", "")}, "log": [], "created": datetime.utcnow().isoformat()}
    _save_job(job)
    _ACTIVE[job_id] = asyncio.create_task(_run(job_id))
    return job_id, ""

def stop(job_id: str):
    job = load_job(job_id)
    if job: job["status"] = "stopping"; _save_job(job)

async def _run_flow(flow: Flow, ctx: StepContext, job_id: str) -> set:
    """Wave-based executor. Returns the set of completed node ids. Subflow nodes recurse."""
    done = set()
    while not flow.is_complete(done):
        job = load_job(job_id)
        if job["status"] == "stopping": return done
        wave = flow.ready(done)
        if not wave: break  # cycle or dangling ref - stop rather than loop forever
        async def run_one(node):
            await ctx.push("step_start", {"node": node.id, "type": node.get("type", ""), "name": node.get("name", "")})
            try:
                if node.get("type") == "subflow":
                    sub = Flow(node.get("flow", {}))
                    await _run_flow(sub, ctx, job_id)
                    result = {k: v for k, v in ctx.scratch.items()}  # subflow shares scratch by design - nested steps see parent context
                else:
                    spec = get_step_type(node.get("type", ""))
                    if not spec: raise RuntimeError(f"unknown step type: {node.get('type')}")
                    result = await spec["fn"](node.get("config", {}), ctx)
                ctx.scratch[node.id] = result
                await ctx.push("step_done", {"node": node.id, "result_keys": list((result or {}).keys())})
            except Exception as e:
                traceback.print_exc()
                await ctx.push("error", {"node": node.id, "message": str(e)})
                raise
            return node.id
        try:
            results = await asyncio.gather(*[run_one(n) for n in wave])
            done.update(results)
        except Exception:
            job = load_job(job_id); job["status"] = "error"; _save_job(job)
            return done
        job = load_job(job_id); job["scratch"] = ctx.scratch; _save_job(job)
    return done

async def _run(job_id: str):
    job = load_job(job_id)
    if not job: return
    job["status"] = "running"; _save_job(job)
    ctx = StepContext(job_id, job["username"], job["scratch"])
    flow = Flow(job["flow"])
    await ctx.push("started", {"nodes_total": len(flow.nodes)})
    done = await _run_flow(flow, ctx, job_id)
    job = load_job(job_id)
    job["status"] = "stopped" if job["status"] == "stopping" else ("error" if not flow.is_complete(done) else "done")
    job["scratch"] = ctx.scratch
    _save_job(job)
    await ctx.push(job["status"], {"scratch_keys": list(ctx.scratch.keys())})
    _ACTIVE.pop(job_id, None)