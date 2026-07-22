""" steps.py — Step type registry + built-in step implementations for ai_manager.
"Any AI operation is a step" - chat completion, a knowledge query, an insert, image generation, future TTS/STT, are all the same shape: async fn(config: dict, ctx: StepContext) -> dict.
The engine never special-cases behavior; it only sequences and reports.
Config values may reference upstream results with {node_id.key} templating, resolved by ctx.resolve() before the step runs - keeps step implementations free of graph-walking logic."""

import httpx, re, json, sys, asyncio, uuid
from pathlib import Path
from tools.ai_manager.connections import get_conn, lightrag_query, lightrag_insert_text, _base
from tools.ai_manager import engine

_STEP_TYPES: dict = {}
ENV: dict = {}
def init(env: dict):
    global ENV
    ENV = env

def register_step_type(name: str, fn, label: str = "", config_schema: dict = None): _STEP_TYPES[name] = {"fn": fn, "label": label or name, "config_schema": config_schema or {}} # label/config_schema are optional UI hints for pipeline builders - the engine itself never reads them.
def get_step_type(name: str) -> dict: return _STEP_TYPES.get(name)
def list_step_types() -> list: return [{"type": k, **{kk: vv for kk, vv in v.items() if kk != "fn"}} for k, v in _STEP_TYPES.items()]

async def step_chat(config: dict, ctx) -> dict:
    """A single non-streaming or streaming chat call against an ollama-compatible connection.
    result_key holds the final text in ctx.scratch for downstream steps/templating."""
    conn = get_conn(config.get("conn_id", ""))
    if not conn: raise RuntimeError("chat step: no connection configured")
    model = config.get("model", "")
    if not model: raise RuntimeError("chat step: no model configured")
    messages = [{"role": "system", "content": ctx.resolve(config["system_prompt"])}] if config.get("system_prompt") else []
    messages.append({"role": "user", "content": ctx.resolve(config.get("user_template", "{input}"))})
    full = ""
    pl = {"model": model, "messages": messages, "stream": True, "options": {"num_ctx": config.get("num_ctx", 8192), "temperature": config.get("temperature", 0.3)}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)) as c:
        async with c.stream("POST", f"{_base(conn)}/api/chat", json=pl) as resp:
            if resp.status_code != 200: return resp #raise RuntimeError(f"chat step: HTTP {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line: continue
                try: chunk = json.loads(line)
                except Exception: continue
                text = chunk.get("message", {}).get("content", "")
                if text: full += text; await ctx.stream(config.get("result_key", "chat_response"), text)
                if chunk.get("done"): break
    result_key = config.get("result_key", "chat_response")
    ctx.scratch[result_key] = full
    return {result_key: full}

async def step_knowledge_query(config: dict, ctx) -> dict:
    conn = get_conn(config.get("conn_id", ""), conn_type="lightrag")
    if not conn: raise RuntimeError("knowledge_query step: no knowledge connection configured")
    q = ctx.resolve(config.get("query_template", "{input}"))
    r = await lightrag_query(conn, q, config.get("mode", "hybrid"))
    result_key = config.get("result_key", "knowledge_response")
    ctx.scratch[result_key] = r.get("response", r.get("error", ""))
    return {result_key: ctx.scratch[result_key]}

async def step_knowledge_insert_text(config: dict, ctx) -> dict:
    conn = get_conn(config.get("conn_id", ""), conn_type="lightrag")
    if not conn: raise RuntimeError("knowledge_insert_text step: no knowledge connection configured")
    text = ctx.resolve(config.get("text_template", "{input}"))
    return await lightrag_insert_text(conn, text, config.get("source_label", ""))

def register_builtins():
    register_step_type("chat", step_chat, "Chat Completion", {"conn_id": "select", "model": "select", "system_prompt": "textarea", "user_template": "textarea", "result_key": "text"})
    register_step_type("knowledge_query", step_knowledge_query, "Knowledge Query", {"conn_id": "select", "mode": "select", "query_template": "textarea", "result_key": "text"})
    register_step_type("knowledge_insert_text", step_knowledge_insert_text, "Knowledge Insert Text", {"conn_id": "select", "text_template": "textarea", "source_label": "text"})
    register_step_type("echo", step_echo, "Echo / Passthrough", {"template": "textarea", "result_key": "text"})
    register_step_type("file_write", step_file_write, "File Write (shadow-staged)", {"fm_root": "text", "shadow_dir": "text", "path": "text", "content_key": "text"})
    register_step_type("python_exec", step_python_exec, "Python Script (deterministic)", {"script_path": "text", "input_template": "textarea", "timeout_s": "number", "result_key": "text"})
    register_step_type("file_write_binary", step_file_write_binary, "File Write Binary (shadow-staged)", {"fm_root":"text","shadow_dir":"text","path":"text","content_key":"text","source_root":"text","result_key":"text"})
    register_step_type("call_pipeline", step_call_pipeline, "Call Another Pipeline", {"pipeline_id": "pipeline_select", "input_template": "textarea", "result_key": "text"})
    register_step_type("foreach_call_pipeline", step_foreach_call_pipeline, "For Each Item, Call Pipeline", {"items_source": "text", "pipeline_id": "pipeline_select", "result_key": "text"})
    register_step_type("chat", step_chat, "Chat Completion", {"conn_id":"select","model":"select","model_ctx":"number","temperature":"number","system_prompt":"textarea","user_template":"textarea","result_key":"text"})
    register_step_type("decision", step_decision, "Decision / Gate", {"conn_id":"select","model":"select","model_ctx":"number","options":"text","system_prompt":"textarea","user_template":"textarea","result_key":"text"})
    register_step_type("branch_on", step_branch_on, "Branch On Decision", {"decision_key":"text","routes_json":"textarea","default_pipeline_id":"pipeline_select","input_template":"textarea","result_key":"text"})
 #   register_step_type("edit_in_place", step_edit_in_place, "Edit In Place (gap-aware)",{"conn_id": "select", "model": "select", "model_ctx": "number", "temperature": "number", "gap_marker": "text", "system_prompt": "textarea", "chunk_tokens": "number", "fm_root": "text", "shadow_dir": "text", "shadow_doc_path": "text", "result_key": "text"})

async def step_echo(config: dict, ctx) -> dict:
    """No-op passthrough: resolves its template against current scratch and returns it under result_key.
    Zero external dependencies - used for engine self-tests and as a manual inspection/breakpoint node when building real pipelines."""
    result_key = config.get("result_key", "echo")
    value = ctx.resolve(config.get("template", "{input}"))
    await ctx.push("echo", {"node_result_key": result_key, "value": value})
    ctx.scratch[result_key] = value
    return {result_key: value}

async def step_file_write(config: dict, ctx) -> dict:
    bi = ENV["tools"]["built_ins"]
    fm_root = config.get("fm_root") or "./data/_common"
    fm = bi.FileManager(fm_root)
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(fm_root) / "_shadow"))
    rel_path = ctx.resolve(config.get("path") or "")
    if not rel_path.strip(): raise RuntimeError("file_write: resolved path is empty - check this node's Location/filename fields")
    content = ctx.scratch.get(config.get("content_key", "answer"), "")
    entry = shadow.stage(rel_path, content, author=f"pipeline:{ctx.job_id}")
    result_key = config.get("result_key", "file_write")
    ctx.scratch[result_key] = {"path": rel_path, "status": entry["status"]}
    return ctx.scratch[result_key]

async def step_python_exec(config: dict, ctx) -> dict:
    """Runs a local script for deterministic processing an LLM shouldn't be doing (bulk file ops, exact formatting, upload workflows).
    script_path only, never inline code - keeps this auditable and grep-able.
    Contract: script receives one resolved argv string; must print a single line of JSON to stdout as its result (or nothing, for pure side-effect scripts).
    Non-zero exit or timeout raises, with stderr surfaced in the error message."""
    script_path = config.get("script_path", "")
    if not script_path or not Path(script_path).is_file(): raise RuntimeError(f"python_exec: script not found: {script_path}")
    arg = ctx.resolve(config.get("input_template", "{input}"))
    timeout_s = int(config.get("timeout_s", 120) or 120)
    proc = await asyncio.create_subprocess_exec(sys.executable, str(script_path), arg,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try: stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill(); await proc.communicate()
        raise RuntimeError(f"python_exec: timed out after {timeout_s}s")
    if proc.returncode != 0: raise RuntimeError(f"python_exec: exit {proc.returncode}\n{stderr.decode(errors='replace')[-1000:]}")
    out = stdout.decode(errors="replace").strip()
    try: parsed = json.loads(out.splitlines()[-1]) if out else {}
    except Exception: parsed = {"raw_stdout": out[-2000:]}
    result_key = config.get("result_key", "python_exec")
    ctx.scratch[result_key] = parsed
    return {result_key: parsed}

async def step_file_write_binary(config: dict, ctx) -> dict:
    bi = ENV["tools"]["built_ins"]
    fm_root = config.get("fm_root") or "./data/_common"
    fm = bi.FileManager(fm_root)
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(fm_root) / "_shadow"))
    src = ctx.scratch.get(config.get("content_key", "image_file"), {})
    source_path = Path(config.get("source_root") or ".") / (src.get("file_name") or "")
    if not source_path.is_file(): raise RuntimeError(f"file_write_binary: source not found: {source_path}")
    rel_path = ctx.resolve(config.get("path") or "")
    if not rel_path.strip(): raise RuntimeError("file_write_binary: resolved path is empty - check this node's Location/filename fields")
    entry = shadow.stage_binary(rel_path, source_path.read_bytes(), author=f"pipeline:{ctx.job_id}")
    result_key = config.get("result_key", "file_write_binary")
    ctx.scratch[result_key] = {"path": rel_path, "status": entry["status"]}
    return ctx.scratch[result_key]

async def step_call_pipeline(config: dict, ctx) -> dict:
    pid = config.get("pipeline_id", "")
    if not pid: raise RuntimeError("call_pipeline: no pipeline selected")
    depth = int(config.get("_call_depth", 0))
    scratch = await engine.run_inline(ctx.username, pid, inputs={"input": ctx.resolve(config.get("input_template") or "{input}")}, depth=depth)
    result_key = config.get("result_key", "call_pipeline")
    ctx.scratch[result_key] = scratch
    return {result_key: scratch}

async def step_foreach_call_pipeline(config: dict, ctx) -> dict:
    """Iterates a list (from scratch, or newline-separated text) and calls the SAME saved pipeline once per item, collecting each sub-run's final scratch.
    This is the loop primitive: e.g. one line per wiki topic -> one pipeline invocation per topic."""
    raw = ctx.scratch.get(config.get("items_source", ""), config.get("items_source", ""))
    if isinstance(raw, str):
        try: items = json.loads(raw)
        except Exception: items = [x.strip() for x in raw.split("\n") if x.strip()]
    else: items = raw if isinstance(raw, list) else []
    if not items: raise RuntimeError("foreach_call_pipeline: items_source resolved to an empty list")
    pid = config.get("pipeline_id", "")
    if not pid: raise RuntimeError("foreach_call_pipeline: no pipeline selected")
    depth = int(config.get("_call_depth", 0))
    results = []
    for i, item in enumerate(items):
        await ctx.progress(f"item {i+1}/{len(items)}: {str(item)[:60]}")
        results.append(await engine.run_inline(ctx.username, pid, inputs={"input": str(item)}, depth=depth))
    result_key = config.get("result_key", "foreach_results")
    ctx.scratch[result_key] = results
    return {result_key: results}

async def _ollama_stream(conn, messages, model, num_ctx, temperature=0.3):
    pl = {"model": model, "messages": messages, "stream": True, "options": {"num_ctx": num_ctx, "temperature": temperature}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)) as c:
        async with c.stream("POST", f"{_base(conn)}/api/chat", json=pl) as resp:
            if resp.status_code != 200: raise RuntimeError(f"HTTP {resp.status_code}")
            async for line in resp.aiter_lines():
                if not line: continue
                try: chunk = json.loads(line)
                except Exception: continue
                text = chunk.get("message", {}).get("content", "")
                if text: yield text
                if chunk.get("done"): break

async def step_chat(config: dict, ctx) -> dict:
    conn = get_conn(config.get("conn_id", ""))
    if not conn: raise RuntimeError("chat step: no connection configured")
    model = config.get("model", "")
    if not model: raise RuntimeError("chat step: no model configured")
    messages = ([{"role": "system", "content": ctx.resolve(config["system_prompt"])}] if config.get("system_prompt") else [])
    messages.append({"role": "user", "content": ctx.resolve(config.get("user_template") or "{input}")})
    full = ""; result_key = config.get("result_key", "chat_response")
    async for piece in _ollama_stream(conn, messages, model, config.get("model_ctx", 8192), config.get("temperature", 0.3)):
        full += piece; await ctx.stream(result_key, piece)
    ctx.scratch[result_key] = full
    return {result_key: full}

async def step_decision(config: dict, ctx) -> dict:
    """Judge/gate: constrains the model to answer with exactly one of a declared set of words (e.g. 'go,redo' or 'accept,merge,retry').
    Pair with branch_on to actually dispatch to a different pipeline based on the answer - this step only produces the decision."""
    conn = get_conn(config.get("conn_id", "")); model = config.get("model", "")
    if not conn or not model: raise RuntimeError("decision: connection/model required")
    options = [o.strip() for o in (config.get("options", "go,redo") or "").split(",") if o.strip()] or ["go", "redo"]
    sys_p = (ctx.resolve(config.get("system_prompt") or "") + f"\n\nRespond with exactly one of these words and nothing else: {', '.join(options)}").strip()
    messages = [{"role": "system", "content": sys_p}, {"role": "user", "content": ctx.resolve(config.get("user_template") or "{input}")}]
    full = ""
    async for piece in _ollama_stream(conn, messages, model, config.get("model_ctx", 8192), 0.0):
        full += piece
    choice = next((o for o in options if o.lower() in full.lower()), options[0])
    result_key = config.get("result_key", "decision")
    ctx.scratch[result_key] = choice
    return {result_key: choice}

async def step_branch_on(config: dict, ctx) -> dict:
    """Reads a scratch value (typically from a decision node) and calls the pipeline mapped to that value.
    routes_json: {"go": "pl_xxx", "redo": "pl_yyy"}.
    This is how a redo loop is expressed in an acyclic graph: 'redo' routes back to a pipeline that (re)does the work, rather than an actual cycle in this DAG - depth-guarded via run_inline."""
    key = config.get("decision_key", "decision")
    value = str(ctx.scratch.get(key, ""))
    try: routes = json.loads(config.get("routes_json", "{}") or "{}")
    except Exception: routes = {}
    pid = routes.get(value) or config.get("default_pipeline_id", "")
    if not pid: raise RuntimeError(f"branch_on: no route configured for decision value '{value}'")
    depth = int(config.get("_call_depth", 0))
    scratch = await engine.run_inline(ctx.username, pid, inputs={"input": ctx.resolve(config.get("input_template") or "{input}")}, depth=depth)
    result_key = config.get("result_key", "branch_result")
    ctx.scratch[result_key] = scratch
    return {result_key: scratch}

async def notstep_edit_in_place(config: dict, ctx) -> dict:
    """True in-place chunk editing: locates a chunk by its exact current text (not by position), replaces just that span via the shadow diff/accept flow, and can detect a gap marker to switch from 'edit existing text' mode to 'write new content to fill the gap' mode, then continues editing past the gap once reached.
    If no marker exists at all, behaves as pure open-ended continuation once the end of existing content is reached - it keeps generating additional chunks until the judge/decision step (chained after this one) reports the goal is met, rather than stopping at end-of-file."""
    pid = config["project_id"]
    doc = _load(pid)  # this project accessor is Tessa's own - registered via extra_config same as other tessa_* steps
    content = doc.get("content", "")
    marker = config.get("gap_marker", "[[GAP]]")
    conn = get_conn(config.get("conn_id", "")); model = config.get("model", "")
    if not conn or not model: raise RuntimeError("edit_in_place: connection/model required")
    chunk_tokens = int(config.get("chunk_tokens", 4000))
    before, _, after = content.partition(marker) if marker in content else (content, "", "")
    chunks = _chunk_text(before, chunk_tokens) if before else []
    edited = []
    for i, chunk in enumerate(chunks):
        await ctx.progress(f"editing chunk {i+1}/{len(chunks)}")
        sys_p = ctx.resolve(config.get("system_prompt") or "")
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content": f"Rewrite this passage in place - same length and content, fix grammar/flow/continuity only:\n\n{chunk}"}]
        full = ""
        async for piece in _ollama_stream(conn, msgs, model, config.get("model_ctx",8192), config.get("temperature",0.3)):
            full += piece
        edited.append(full.strip() or chunk)
    if marker in content and after.strip():
        # gap exists with real continuation text after it - generate filler chunks until a decision step (chained next in the pipeline) judges the gap closed. This step produces ONE filler pass per run; loop via branch_on->redo for multiple passes.
        sys_p = ctx.resolve(config.get("system_prompt") or "")
        fill_prompt = f"Continue the story to bridge this gap. End of text before the gap:\n\n{edited[-1][-1500:] if edited else before[-1500:]}\n\nText that must follow after your bridge:\n\n{after[:1500]}"
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content":fill_prompt}]
        full = ""
        async for piece in _ollama_stream(conn, msgs, model, config.get("model_ctx",8192), config.get("temperature",0.5)):
            full += piece
        new_content = "\n\n".join(edited) + "\n\n" + full.strip() + "\n\n" + after
    elif marker in content:
        # marker present but nothing after it - open-ended: generate one additional chunk past the existing content. Chain with decision/branch_on to keep generating.
        sys_p = ctx.resolve(config.get("system_prompt") or "")
        fill_prompt = f"Continue this story toward its stated goal. Existing text ends:\n\n{(edited[-1] if edited else before)[-1500:]}"
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content":fill_prompt}]
        full = ""
        async for piece in _ollama_stream(conn, msgs, model, config.get("model_ctx", 8192), config.get("temperature",0.5)):
            full += piece
        new_content = "\n\n".join(edited) + "\n\n" + full.strip()
    else:
        new_content = "\n\n".join(edited) or content
    bi = ENV["tools"]["built_ins"]
    fm = bi.FileManager(config.get("fm_root") or "./data/_common")
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(config.get("fm_root") or "./data/_common") / "_shadow"))
    entry = shadow.stage(config.get("shadow_doc_path", f"tessa_edits/{pid}.md"), new_content, author=f"pipeline:{ctx.job_id}")
    result_key = config.get("result_key", "edit_in_place")
    ctx.scratch[result_key] = {"status": entry["status"], "gap_remaining": bool(marker in content and not after.strip())}
    return ctx.scratch[result_key]