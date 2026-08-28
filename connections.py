"""conntections.py
Shared utilities for all ai tools conenctions.
Not a sub-module (no subdirectory/router) - import directly.
"""
import json, httpx, time
from pathlib import Path

_QUERY_CACHE: dict = {}
_MISSING = object()
_TEMPLATE_CACHE: dict = {}
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")
_TMPL_DIR = Path("./tools/ai_manager/_connections")   # provider schema profiles - code-shipped
CONN_DIR  = Path("./data/ai_manager/_connections")    # saved connection instances - tool-owned data, cross-module
CONN_DIR.mkdir(parents=True, exist_ok=True)

def list_templates() -> dict: return {f.stem: json.loads(f.read_text()) for f in sorted(_TMPL_DIR.glob("*.json"))}
def save_conn(cid: str, data: dict): (CONN_DIR / f"{cid}.json").write_text(json.dumps(data, indent=2))
def delete_conn(cid: str): (CONN_DIR / f"{cid}.json").unlink(missing_ok=True)
def load_conn_raw(cid: str) -> dict | None:
    p = CONN_DIR / f"{cid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_conns(conn_type="ollama", get_all = False) -> list:
    out = []
    for f in sorted(CONN_DIR.glob("*.json")):
        try:
            c = json.loads(f.read_text())
            if not get_all and conn_type and c.get("connection_type") != conn_type: continue
            c["_id"] = f.stem
            out.append(c)
        except: pass
    return out

def get_conn(conn_id="", conn_type="ollama"):
    """Explicit id always resolves regardless of type - a user's actual selection shouldn't be filtered out by whatever default type a caller happens to use. Empty id falls back to 'first of conn_type'."""
    if conn_id: return next((c for c in list_conns(get_all=True) if c["_id"] == conn_id), None)
    return next(iter(list_conns(conn_type)), None)

def _base(conn) -> str:
    v = conn.get("values", {})
    host = str(v.get("host", "127.0.0.1")).strip().rstrip("/")
    if "://" in host: return f"{host}{v.get('base_path','')}"  # full origin pasted directly
    has_port = ":" in host
    scheme = "https" if v.get("tls") else "http"
    port = "" if has_port else (f":{v['port']}" if v.get("port") else "")
    base_path = str(v.get("base_path", "")).strip()
    if base_path and not base_path.startswith("/"): base_path = "/" + base_path
    return f"{scheme}://{host}{port}{base_path}"

async def list_models_async(conn) -> list:
    if not conn: return []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)) as c:
            r = await c.get(f"{_base(conn)}/api/tags")
            return sorted(m["name"] for m in r.json().get("models",[])) if r.status_code==200 else []
    except Exception:
        return []  # unreachable backend is an external failure, not a program bug - caller shows an empty list and lets the user pick another connection

def list_models_sync(conn) -> list:
    if not conn: return []
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)) as c:
            r = c.get(f"{_base(conn)}/api/tags")
            return sorted(m["name"] for m in r.json().get("models",[])) if r.status_code==200 else []
    except Exception:
        return []

def tok_estimate(text: str) -> int: return max(1, len(str(text)) // 4) # Fast 4-chars-per-token estimate for English prose.

async def tok_count_api(conn, model: str, text: str) -> int:
    """Accurate count via ollama /api/embed prompt_eval_count. Falls back to estimate. Use sparingly - makes an API round-trip."""
    if not conn or not model: return tok_estimate(text)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=3.0)) as c:
            r = await c.post(f"{_base(conn)}/api/embed", json={"model": model, "input": text})
            if r.status_code == 200: return r.json().get("prompt_eval_count") or tok_estimate(text)
    except: pass
    return tok_estimate(text)

def conn_opts_html(selected_id="", conn_type="ollama") -> str: return "".join(f'<option value="{c["_id"]}" {"selected" if c["_id"]==selected_id else ""}>{c.get("display_name",c["_id"])}</option>' for c in list_conns(conn_type)) or '<option value="">No connections configured</option>'

def model_opts_html(conn_id="", selected_model="") -> str:
    conn = get_conn(conn_id)
    models = list_models_sync(conn) if conn else []
    opts = "".join(f'<option value="{m}" {"selected" if m==selected_model else ""}>{m}</option>' for m in models)
    return opts or f'<option value="{selected_model}">{selected_model or "-- select connection first --"}</option>'

def is_prefix_breaking_change(conn: dict, changed_field: str) -> bool: return changed_field in _conn_profile(conn).get("prefix_breaking_fields", []) # True if changing this field on an active conversation invalidates the model's cached KV prefix for that connection type. A caller should warn before applying such a change to a conversation with existing history, per this connection type's own declared profile - not guessed at per-module.

# --- Universal LLM Generation (Template-Driven Provider Interface) ---
# Each connection_type's JSON profile (modules/ai_tools/_connections/{type}.json) declares its own wire format under payload_template/stream_format/content_path - stream_llm never hardcodes a provider.
# Adding a new backend (vLLM, LM Studio, a cloud API) is a JSON file drop, not a code change here.
def _conn_profile(conn: dict) -> dict:
    ctype = conn.get("connection_type", "ollama")
    if ctype in _TEMPLATE_CACHE: return _TEMPLATE_CACHE[ctype]
    p = _TMPL_DIR / f"{ctype}.json"
    profile = json.loads(p.read_text()) if p.exists() else {}
    _TEMPLATE_CACHE[ctype] = profile
    return profile

def _resolved_options(profile: dict, kwargs: dict) -> dict:
    """None (missing OR explicitly blank) always means 'use this schema's own default' - never silently becomes 0."""
    out = {}
    for key, spec in profile.get("options_schema", {}).items():
        val = kwargs.get(key)
        if val is None: val = spec.get("default")
        if isinstance(val, (int, float)) and spec.get("type") in ("integer", "float"):
            if "min" in spec: val = max(spec["min"], val)
            if "max" in spec: val = min(spec["max"], val)
        out[key] = val
    return out

def _render_template(node, params: dict):
    """Recursively substitutes '{param}' placeholders against params - only keys present in params are filled in;
    a placeholder with no matching param is dropped from the payload rather than sent literally, so provider templates can offer optional fields (max_tokens, top_p, ...) without every caller needing to supply them."""
    if isinstance(node, dict): return {k: v for k, v in ((k, _render_template(v, params)) for k, v in node.items()) if v is not _MISSING}
    if isinstance(node, list): return [_render_template(v, params) for v in node]
    if isinstance(node, str) and node.startswith("{") and node.endswith("}"): return params[node[1:-1]] if node[1:-1] in params else _MISSING
    return node

def _get_nested_value(data, path: str):
    """Dot-notation lookup (e.g. 'choices.0.delta.content') for pulling a text chunk out of an arbitrary provider response shape."""
    if not path: return ""
    for key in path.split("."):
        if isinstance(data, list) and key.isdigit(): data = data[int(key)] if int(key) < len(data) else None
        elif isinstance(data, dict): data = data.get(key)
        else: return ""
        if data is None: return ""
    return data

async def stream_llm(conn: dict, messages: list, model: str, think = False, **kwargs):
    """Agnostic chat stream. Yields (text, thinking) tuples per chunk - thinking is "" for providers/calls that don't produce it.
    think=True is honored only if the connection's profile declares supports_thinking;
    otherwise it's silently dropped with a console warning, never an error - a provider lacking a capability is a gap in that provider, not a reason to remove the capability for providers that have it."""
    profile = _conn_profile(conn)
    chat_ep = profile.get("endpoints", {}).get("chat", {})
    if not chat_ep: raise RuntimeError(f"stream_llm: connection_type '{conn.get('connection_type')}' has no endpoints.chat")
    supports_thinking = profile.get("supports_thinking", False)
    if think and not supports_thinking: print(f"[stream_llm] '{conn.get('connection_type')}' has no supports_thinking - think={think!r} ignored for this call")
    options = _resolved_options(profile, kwargs)
    payload = _render_template(chat_ep.get("body", {}), {"model": model, "messages": messages, "think": (think if supports_thinking else False), "options": options, **options})
    url = _base(conn) + chat_ep.get("path", "/api/chat")
    headers = {}
    api_key = conn.get("values", {}).get("api_key", "")
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    parser = profile.get("response_parser", {"stream_format": "jsonl", "content_path": "message.content"})
    done_sentinel = profile.get("status_map", {}).get("stream_done_sentinel", {})
    timeout_s = float(conn.get("values", {}).get("timeout_s", 3000))
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=10.0, pool=10.0)) as c:
        async with c.stream(chat_ep.get("method","POST"), url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"connection error {resp.status_code} calling {conn.get('_id','?')} model '{model}': {body.decode(errors='replace')[:500]}")
            async for line in resp.aiter_lines():
                if not line: continue
                if parser.get("stream_format") == "sse":
                    if not line.startswith("data: "): continue
                    line = line[6:]
                    if line.strip() == "[DONE]": break
                try: chunk = json.loads(line)
                except json.JSONDecodeError: continue
                text = _get_nested_value(chunk, parser.get("content_path",""))
                thinking = _get_nested_value(chunk, parser.get("thinking_path","")) if supports_thinking else ""
                if text or thinking: yield text, thinking
                if done_sentinel and chunk.get(done_sentinel.get("key","done")) == done_sentinel.get("value", True): break

# --- LightRAG (config-driven - see tools/ai_manager/_connections/lightrag.json) ---
# No LightRAG-version-specific shape lives in this file. Every operation reads its method/path/body template from the connection's own JSON profile and renders it generically, exactly like stream_llm does for chat completions.
# Adjusting for a different LightRAG version/fork/API-compatible knowledge backend is a JSON edit, never a code change here - these functions describe a KNOWLEDGE BASE operation, not a specific server's wire format.

def _lightrag_profile(conn: dict) -> dict:
    p = _TMPL_DIR / "lightrag.json"
    return json.loads(p.read_text()) if p.exists() else {}

def _fmt_path(path: str, params: dict) -> str:
    for k, v in params.items(): path = path.replace("{" + str(k) + "}", str(v))
    return path

async def _lightrag_call(conn, endpoint_name: str, params: dict = None, extra_body: dict = None, file_bytes: bytes = None, file_name: str = "") -> dict:
    profile = _lightrag_profile(conn)
    ep = profile.get("endpoints", {}).get(endpoint_name, {})
    if not ep: return {"error": f"lightrag profile has no '{endpoint_name}' endpoint declared - add it to lightrag.json"}
    params = params or {}
    method, path = ep.get("method", "GET"), _fmt_path(ep.get("path", "/"), params)
    fallback = _fmt_path(ep.get("fallback_path", ""), params) if ep.get("fallback_path") else ""
    body = _render_template(ep.get("body", {}), params) if ep.get("body") else None
    if body is not None and extra_body: body = {**body, **extra_body}
    read_timeout = float(conn.get("values", {}).get("timeout_s", 2400))
    async def _do(c, p):
        if file_bytes is not None: return await c.request(method, _base(conn)+p, files={ep.get("file_field","file"): (file_name, file_bytes)})
        if method == "GET": return await c.request(method, _base(conn)+p, params=body)
        return await c.request(method, _base(conn)+p, json=body)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=read_timeout, write=60.0, pool=20.0)) as c:
            r = await _do(c, path)
            if r.status_code == 404 and fallback: r = await _do(c, fallback)
            if r.status_code not in (200, 204): return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
            return r.json() if r.content else {"ok": True}
    except Exception as e: return {"error": str(e)}

def lightrag_query_options_schema(conn) -> dict:
    """UI callers (Kimi) render this dynamically as query-time toggles/fields - a new option is a JSON edit, never a new checkbox hand-added to kimi.py."""
    return _lightrag_profile(conn).get("query_options_schema", {})

async def lightrag_health(conn) -> dict:
    r = await _lightrag_call(conn, "health")
    return {"ok": "error" not in r, "detail": r.get("error", r)}

async def lightrag_query_stream(conn, query: str, mode: str = "mix", **extra):
    """NDJSON streaming query against /query/stream per LightRAG's documented contract.
    Yields (chunk_text, references, error) - references arrive on the first line if requested, error terminates iteration."""
    profile = _lightrag_profile(conn)
    ep = profile.get("endpoints", {}).get("query_stream", {})
    if not ep: yield "", None, "lightrag profile has no 'query_stream' endpoint declared"; return
    schema = lightrag_query_options_schema(conn)
    opts = {k: extra.pop(k, spec.get("default")) for k, spec in schema.items()}
    body = {"query": query, "mode": mode, "stream": True, **{k:v for k,v in opts.items() if v is not None}, **{k:v for k,v in extra.items() if v is not None}}
    read_timeout = float(conn.get("values", {}).get("timeout_s", 2400))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=read_timeout, write=60.0, pool=20.0)) as c:
            async with c.stream("POST", _base(conn) + ep.get("path", "/query/stream"), json=body) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    yield "", None, f"HTTP {resp.status_code}: {text.decode(errors='replace')[:300]}"
                    return
                async for line in resp.aiter_lines():
                    if not line.strip(): continue
                    try: data = json.loads(line)
                    except json.JSONDecodeError: continue
                    if "error" in data: yield "", None, data["error"]; return
                    if "references" in data: yield "", data["references"], None
                    if "response" in data: yield data["response"], None, None
    except Exception as e:
        yield "", None, str(e)

async def lightrag_query(conn, query: str, mode: str = "hybrid", **extra) -> dict:
    schema = lightrag_query_options_schema(conn)
    opts = {k: extra.pop(k, spec.get("default")) for k, spec in schema.items()}
    extra_body = {**{k:v for k,v in opts.items() if v is not None}, **{k:v for k,v in extra.items() if v is not None}}
    r = await _lightrag_call(conn, "query", {"query": query, "mode": mode}, extra_body=extra_body)
    if _lr_failed(r): return r
    parser = _lightrag_profile(conn).get("endpoints",{}).get("query",{}).get("response_parser", {"content_path":"response"})
    return {"response": _get_nested_value(r, parser.get("content_path","response")) or r.get("response",""), "raw": r}

async def lightrag_insert_text(conn, text: str, source: str = "") -> dict: return await _lightrag_call(conn, "insert_text", {"text": text, "source": source or "manual entry"})
async def lightrag_insert_file(conn, filename: str, content: bytes) -> dict: return await _lightrag_call(conn, "insert_file", file_bytes=content, file_name=filename)
async def lightrag_list_documents(conn) -> dict: return await _lightrag_call(conn, "list_documents")
async def lightrag_delete_document(conn, doc_id: str) -> dict: return await _lightrag_call(conn, "delete_document", {"doc_id": doc_id})

async def lightrag_clear_all(conn) -> dict:
    r = await _lightrag_call(conn, "clear_all")
    return {"ok": True} if "error" not in r else r

async def lightrag_query_cached(conn, query: str, mode: str = "hybrid", ttl: int = 300, **extra) -> dict:
    key = (conn.get("_id", id(conn)), mode, query, tuple(sorted(extra.items())))
    now = time.time()
    hit = _QUERY_CACHE.get(key)
    if hit and now - hit[0] < ttl: return hit[1]
    result = await lightrag_query(conn, query, mode, **extra)
    _QUERY_CACHE[key] = (now, result)
    return result

async def lightrag_context_block(conn, query: str, mode: str = "hybrid", label: str = "Knowledge") -> str:
    r = await lightrag_query_cached(conn, query, mode)
    text = r.get("response", "").strip()
    return f"[{label}]\n{text}\n" if text else ""

def _parse_graph_response(data) -> tuple:
    """Normalizes the two response shapes seen across LightRAG versions/forks - {"nodes":[...],"edges":[...]} or a bare list of label strings/objects. Extend here (the one shared place), never per-caller, if a new shape appears."""
    if isinstance(data, dict): return data.get("nodes", []), data.get("edges", [])
    if isinstance(data, list): return [{"id": i, "label": i} if isinstance(i, str) else i for i in data], []
    return [], []

async def lightrag_graph_dot(conn, limit: int = 1000, label: str = "*") -> str:
    data = await _lightrag_call(conn, "graph", {"limit": limit, "label": label, "max_depth": 3, "max_nodes": limit})
    if _lr_failed(data): print(f"[LightRAG] graph fetch error: {data['error']}"); return ""
    nodes, edges = _parse_graph_response(data)
    if not nodes: return ""
    lines = ["digraph G {", '  graph [layout="fdp", bgcolor="black", splines=curved, overlap=false, K=0.7, nodesep=0.8];', '  node [shape=circle, style=filled, fontname="Helvetica", fontsize=9, fixedsize=true, width=0.6, height=0.6, color="#2c3e50", fillcolor="#3498db", fontcolor="white"];', '  edge [color="#444444", penwidth=1.0, arrowsize=0.5];']
    for n in nodes[:limit]:
        label = str(n.get("label") or n.get("id") or n.get("entity_name") or "?").replace('"', "'")
        lines.append(f'  "{str(n.get("id", label))}" [xlabel="{label}", label=""];')
    for e in edges[:limit*2]:
        src, dst = e.get("source", e.get("from", e.get("src_id"))), e.get("target", e.get("to", e.get("tgt_id")))
        if src and dst: lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)

async def lightrag_list_entities(conn, limit: int = 1000) -> list:
    data = await _lightrag_call(conn, "graph", {"limit": limit})
    if _lr_failed(data): print(f"[LightRAG] graph fetch error: {data['error']}"); return []
    nodes, _ = _parse_graph_response(data)
    return nodes

def _lr_failed(r) -> bool:
    """True only on a REAL failure. Only our own _lightrag_call ever sets 'error' to a truthy message (on an exception or non-2xx status) - a response dict that merely contains a falsy 'error' key (some LightRAG versions include this as a no-op success sentinel) is not a failure and must not be treated as one."""
    return isinstance(r, dict) and bool(r.get("error"))

# --- Flux2 Image Pipeline (text encoder + image generator nodes) ---
# connection_type="flux2_text"  - prompt -> embeds.npy in a shared vault (see modules/ai_tools/_connections/flux2_text.json)
# connection_type="flux2_image" - embeds + params -> PNG/GIF (see modules/ai_tools/_connections/flux2_image.json)
# Both nodes are synchronous, single-request-at-a-time by design (one GPU each). These helpers make one call and report what happened - they do not retry or queue. Callers own their own serialization.

async def flux2_health(conn) -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)) as c:
            r = await c.get(f"{_base(conn)}/health")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_encode(conn, prompt: str, job_id: str = None, max_sequence_length: int = 512, hard_truncate: bool = False, force_recompute: bool = False, include_negative: bool = False) -> dict:
    """Blocking on the encoder node (subprocess per prompt, up to ~30min) - call from a background task, never inline in a request handler."""
    payload = {"prompt": prompt, "job_id": job_id, "max_sequence_length": max_sequence_length, "hard_truncate": hard_truncate, "force_recompute": force_recompute, "include_negative": include_negative}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=9000.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/encode", json=payload)
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e: return {"error": str(e)}

async def flux2_generate(conn, payload: dict) -> dict:
    """Blocking on the image node (single GPU, serializes naturally) - caller's queue worker holds this until it returns."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=3600.0, write=30.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/generate", json=payload)
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e: return {"error": str(e)}

async def flux2_generate_sequence(conn: dict, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=3600.0, write=30.0, pool=10.0)) as c:
        r = await c.post(f"{_base(conn)}/generate/sequence", json=payload)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}

async def flux2_system_status(conn) -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)) as c:
            r = await c.get(f"{_base(conn)}/system/status")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_system_load(conn, model_path: str, vae_path: str) -> dict:
    """/system/load is a no-op if the engine already has something loaded - unload first to switch models."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/system/load", params={"model_path": model_path, "vae_path": vae_path})
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_system_unload(conn) -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/system/unload")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_system_stop(conn) -> dict:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/system/stop")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

def list_loras(lora_dir, exts=(".safetensors", ".pt", ".ckpt")) -> list: return sorted(f.name for f in Path(lora_dir).glob("*") if f.is_file() and f.suffix.lower() in exts) if Path(lora_dir).exists() else []

# --- Machine / Node Profiles ---

def set_machine_profile(conn_id: str, profile: dict):
    conn = load_conn_raw(conn_id)
    if conn: conn["machine_profile"] = profile; save_conn(conn_id, conn)

def get_machine_profile(conn_id: str) -> dict:
    conn = load_conn_raw(conn_id) or {}
    return conn.get("machine_profile", {"shares_hardware_with": [], "tags": [], "weight_speed": 1.0, "weight_quality": 1.0, "max_concurrent": 1})

def rank_connections(conn_type: str, tags: list = None, priority: str = "balanced") -> list:
    """priority: speed | balanced | quality - the three modes from the resource-pool design."""
    tags = set(tags or [])
    scored = []
    for c in list_conns(conn_type):
        prof = c.get("machine_profile", {})
        tag_hit = len(tags & set(prof.get("tags", [])))
        w = {"speed": prof.get("weight_speed",1.0), "quality": prof.get("weight_quality",1.0), "balanced": (prof.get("weight_speed",1.0)+prof.get("weight_quality",1.0))/2}[priority]
        scored.append((tag_hit * 10 + w, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]