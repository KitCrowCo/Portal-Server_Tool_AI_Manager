""" steps.py — Step type registry + built-in step implementations for ai_manager.
"Any AI operation is a step" - chat completion, a knowledge query, an insert, image generation, future TTS/STT, are all the same shape: async fn(config: dict, ctx: StepContext) -> dict.
The engine never special-cases behavior; it only sequences and reports.
Config values may reference upstream results with {node_id.key} templating, resolved by ctx.resolve() before the step runs - keeps step implementations free of graph-walking logic."""

import httpx, re, json
from pathlib import Path
from tools.ai_manager.connections import get_conn, lightrag_query, lightrag_insert_text, _base

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

# steps.py — add to imports
import sys, asyncio, uuid   # uuid was already used in step_image_generate but never imported - real bug, fixed here too

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