"""conntections.py
Shared utilities for all ai tools conenctions.
Not a sub-module (no subdirectory/router) - import directly.
"""
import json, httpx, time, re, asyncio, hashlib
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
    """Model names from the profile's list_models endpoint, read with its model_list_parser (root_key, name_field); Ollama's /api/tags shape when the profile names neither."""
    if not conn: return []
    profile = _conn_profile(conn)
    path, mp = profile.get("endpoints", {}).get("list_models", {}).get("path", "/api/tags"), profile.get("model_list_parser") or {}
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)) as c:
            r = c.get(f"{_base(conn)}{path}", headers=_auth_headers(conn))
            return sorted(str(m[mp.get("name_field", "name")]) for m in r.json().get(mp.get("root_key", "models"), [])) if r.status_code==200 else []
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
    """Recursively substitutes placeholders against params. A string that is exactly one '{key}' returns the raw value (lists/dicts/numbers stay typed); a string with embedded '{key}' parts is formatted in place.
    Any placeholder with no matching param drops the whole field, so profiles can declare optional fields without every caller supplying them."""
    if isinstance(node, dict): return {k: v for k, v in ((k, _render_template(v, params)) for k, v in node.items()) if v is not _MISSING}
    if isinstance(node, list): return [_render_template(v, params) for v in node]
    if isinstance(node, str):
        keys = re.findall(r"\{(\w+)\}", node)
        if not keys: return node
        if any(k not in params for k in keys): return _MISSING
        if re.fullmatch(r"\{\w+\}", node): return params[keys[0]]
        return re.sub(r"\{(\w+)\}", lambda m: str(params[m.group(1)]), node)
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

def connection_capabilities(conn: dict) -> dict:
    """What this connection's own profile declares it can do - checked before ever routing a request to it. A missing/false key always means 'not supported', never assumed true."""
    return _conn_profile(conn).get("capabilities", {})

def _auth_headers(conn: dict) -> dict:
    """The connection's api_key in the header its profile names ("auth_header": {"name", "format"}), Authorization: Bearer <key> when the profile names none. No key, no header."""
    key, spec = conn.get("values", {}).get("api_key"), _conn_profile(conn).get("auth_header") or {"name": "Authorization", "format": "Bearer {key}"}
    return {spec["name"]: spec["format"].replace("{key}", key)} if key else {}

def _norm_tool_call(tc: dict) -> dict:
    """Ollama sends function.arguments as a dict, OpenAI-shaped providers as a JSON string - every caller gets {id, name, arguments: dict}."""
    fn = tc.get("function", tc)
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try: args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError: args = {"_raw": args}
    return {"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args if isinstance(args, dict) else {"_raw": args}}

THINK_LEVELS = ("low", "medium", "high", "max")

def _think_value(v):
    """What providers accept for think: an effort level or a bool. Settings store "Off" as "" (and older ones "true" / "false"), which Ollama rejects with 400 - anything else becomes a bool."""
    return v if isinstance(v, bool) or v in THINK_LEVELS else str(v).strip().lower() in ("true", "1", "on", "yes")

async def stream_llm_events(conn: dict, messages: list, model: str, think = False, tools: list = None, **kwargs):
    """Agnostic chat stream. Yields {"text", "thinking", "tool_calls"} per chunk - empty values for whatever a chunk doesn't carry.
    think=True is honored only if the connection's profile declares supports_thinking; otherwise it's dropped with a console warning, never an error - a provider lacking a capability is a gap in that provider, not a reason to remove the capability for providers that have it.
    tools (OpenAI/Ollama function list) is sent only when given AND the profile's chat body declares a {tools} placeholder; returned calls are read from response_parser.tool_calls_path and normalized by _norm_tool_call."""
    profile = _conn_profile(conn)
    chat_ep = profile.get("endpoints", {}).get("chat", {})
    if not chat_ep: raise RuntimeError(f"""stream_llm: connection_type '{conn.get('connection_type')}' has no endpoints.chat""")
    supports_thinking, think = profile.get("supports_thinking", False), _think_value(think)
    if think and not supports_thinking: print(f"""[stream_llm] '{conn.get('connection_type')}' has no supports_thinking - think={think!r} ignored for this call""")
    options = _resolved_options(profile, kwargs)
    think = profile.get("think_map", {}).get(str(think if supports_thinking else False).lower(), think if supports_thinking else False)   # a profile can map on / off / a level to its provider's own value (OpenAI-style reasoning_effort: off -> "none")
    payload = _render_template(chat_ep.get("body", {}), {"model": model, "messages": messages, "think": think, "options": options, **options, **({"tools": tools} if tools else {})})
    if tools and "tools" not in payload: print(f"""[stream_llm] '{conn.get('connection_type')}' chat body has no {{tools}} placeholder - tools not sent for this call""")
    url = _base(conn) + chat_ep.get("path", "/api/chat")
    parser = profile.get("response_parser", {"stream_format": "jsonl", "content_path": "message.content"})
    done_sentinel = profile.get("status_map", {}).get("stream_done_sentinel", {})
    timeout_s = float(conn.get("values", {}).get("timeout_s", 3000))
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=10.0, pool=10.0)) as c:
        async with c.stream(chat_ep.get("method","POST"), url, json=payload, headers=_auth_headers(conn)) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"""connection error {resp.status_code} calling {conn.get('_id','?')} model '{model}': {body.decode(errors='replace')[:500]}""")
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
                calls = _get_nested_value(chunk, parser.get("tool_calls_path","")) if tools else ""
                if text or thinking or calls: yield {"text": text or "", "thinking": thinking or "", "tool_calls": [_norm_tool_call(tc) for tc in calls] if isinstance(calls, list) else []}
                if done_sentinel and chunk.get(done_sentinel.get("key","done")) == done_sentinel.get("value", True): break

async def stream_llm(conn: dict, messages: list, model: str, think = False, **kwargs):
    """(text, thinking) tuple view over stream_llm_events, for every caller that doesn't use tool calls."""
    async for ev in stream_llm_events(conn, messages, model, think, **kwargs):
        if ev["text"] or ev["thinking"]: yield ev["text"], ev["thinking"]

def has_capability(conn: dict, capability: str, model: str = "") -> bool:
    """True/False capabilities are static. A dict capability checks its 'models' sublayer -
    absent model or empty sublayer defaults to capable, so declaring the sublayer costs nothing until you actually start populating per-model exceptions."""
    cap = _conn_profile(conn).get("capabilities", {}).get(capability)
    if isinstance(cap, dict):
        models = cap.get("models", {})
        return models.get(model, True) if model else True
    return bool(cap)

def _capability_request(conn: dict, capability: str, payload: dict) -> tuple:
    """(method, url, body) for one capability from the connection's profile, or (None, None, error text) - shared by call_capability and call_capability_sync."""
    if not has_capability(conn, capability): return None, None, f"""connection '{conn.get('_id','?')}' ({conn.get('connection_type','?')}) does not declare capability '{capability}'"""
    ep = _conn_profile(conn).get("endpoints", {}).get(capability, {})
    if not ep: return None, None, f"""profile declares '{capability}' but has no endpoints.{capability} - profile misconfigured"""
    return ep.get("method", "POST"), _base(conn) + ep.get("path", "/"), _render_template(ep.get("body", {}), payload)

async def call_capability(conn: dict, capability: str, payload: dict, timeout_s: float = None) -> dict:
    """Generic dispatcher for any non-streaming capability a connection's profile declares - image/video generation, context-cache save/load, LoRA management, or anything not yet imagined.
    Same config-over-code shape as stream_llm; never named after a capability or backend."""
    method, url, body = _capability_request(conn, capability, payload)
    if not method: return {"error": body}
    read_timeout = timeout_s if timeout_s is not None else float(conn.get("values", {}).get("timeout_s", 3600))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=30.0, pool=10.0)) as c:
            r = await c.request(method, url, json=body, headers=_auth_headers(conn))
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e: return {"error": str(e)}

def call_capability_sync(conn: dict, capability: str, payload: dict = None, timeout_s: float = 10.0) -> dict:
    """Blocking sibling of call_capability for quick lookups made while rendering (option lists such as a node's voices or languages). Short default timeout - never use it for generation."""
    method, url, body = _capability_request(conn, capability, payload or {})
    if not method: return {"error": body}
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=3.0, read=timeout_s, write=5.0, pool=3.0)) as c:
            r = c.request(method, url, json=body, headers=_auth_headers(conn))
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e: return {"error": str(e)}

def conns_matching(kind: str) -> list: return [c for c in list_conns(get_all=True) if c.get("connection_type") == kind or has_capability(c, kind)] #kind is a connection_type OR a capability name - callers that only care what a connection can DO should pass the capability.

async def stream_capability(conn: dict, capability: str, payload: dict):
    """Streaming sibling of call_capability for endpoints that answer with NDJSON progress lines (model pulls, long jobs). Yields one parsed dict per line; a failure yields a single {"error": ...} dict instead of raising."""
    profile = _conn_profile(conn)
    if not has_capability(conn, capability): yield {"error": f"connection '{conn.get('_id','?')}' does not declare capability '{capability}'"}; return
    ep = profile.get("endpoints", {}).get(capability, {})
    if not ep: yield {"error": f"profile declares '{capability}' but has no endpoints.{capability}"}; return
    body = _render_template(ep.get("body", {}), payload)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=float(conn.get("values", {}).get("timeout_s", 3600)), write=30.0, pool=10.0)) as c:
            async with c.stream(ep.get("method", "POST"), _base(conn) + ep.get("path", "/"), json=body, headers=_auth_headers(conn)) as r:
                if r.status_code != 200: yield {"error": f"HTTP {r.status_code}: {(await r.aread()).decode(errors='replace')[:300]}"}; return
                async for line in r.aiter_lines():
                    if not line.strip(): continue
                    try: yield json.loads(line)
                    except json.JSONDecodeError: continue
    except Exception as e: yield {"error": str(e)}

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
        if file_bytes is not None: return await c.request(method, _base(conn)+p, files={ep.get("file_field","file"): (file_name, file_bytes)}, headers=_auth_headers(conn))
        if method == "GET": return await c.request(method, _base(conn)+p, params=body, headers=_auth_headers(conn))
        return await c.request(method, _base(conn)+p, json=body, headers=_auth_headers(conn))
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

def lightrag_query_defaults(conn) -> dict:
    """Query options in effect for one knowledge group: the profile's schema defaults, overlaid with the group's own saved defaults (conn["query_defaults"], set from Kimi).
    Every caller (Kimi, knowledge nodes, context blocks) gets the same per-group behaviour, e.g. rerank on for one group and off for another."""
    return {**{k: spec.get("default") for k, spec in lightrag_query_options_schema(conn).items()}, **(conn.get("query_defaults") or {})}

def base_url(conn) -> str: return _base(conn)   # for links to a server's own pages (e.g. LightRAG's WebUI)

async def lightrag_op(conn, name: str, **params) -> dict:
    """Any operation declared under endpoints in the connection's profile, by name - pipeline_status, track_status (track_id=...), reprocess_failed, cancel_pipeline, status_counts, graph_labels, openapi.
    A server without that route answers with its own HTTP error, returned as {"error": ...}."""
    return await _lightrag_call(conn, name, params)

async def lightrag_health(conn) -> dict:
    r = await _lightrag_call(conn, "health")
    return {"ok": "error" not in r, "detail": r.get("error", r)}

async def lightrag_query_stream(conn, query: str, mode: str = "mix", **extra):
    """NDJSON streaming query against /query/stream per LightRAG's documented contract.
    Yields (chunk_text, references, error) - references arrive on the first line if requested, error terminates iteration."""
    profile = _lightrag_profile(conn)
    ep = profile.get("endpoints", {}).get("query_stream", {})
    if not ep: yield "", None, "lightrag profile has no 'query_stream' endpoint declared"; return
    opts = {k: extra.pop(k, v) for k, v in lightrag_query_defaults(conn).items()}
    body = {"query": query, "mode": mode, "stream": True, **{k:v for k,v in opts.items() if v is not None}, **{k:v for k,v in extra.items() if v is not None}}
    read_timeout = float(conn.get("values", {}).get("timeout_s", 2400))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=read_timeout, write=60.0, pool=20.0)) as c:
            async with c.stream("POST", _base(conn) + ep.get("path", "/query/stream"), json=body, headers=_auth_headers(conn)) as resp:
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
    opts = {k: extra.pop(k, v) for k, v in lightrag_query_defaults(conn).items()}
    extra_body = {**{k:v for k,v in opts.items() if v is not None}, **{k:v for k,v in extra.items() if v is not None}}
    r = await _lightrag_call(conn, "query", {"query": query, "mode": mode}, extra_body=extra_body)
    if _lr_failed(r): return r
    parser = _lightrag_profile(conn).get("endpoints",{}).get("query",{}).get("response_parser", {"content_path":"response"})
    return {"response": _get_nested_value(r, parser.get("content_path","response")) or r.get("response",""), "raw": r}

async def lightrag_insert_text(conn, text: str, source: str = "", pieces: list = None) -> dict:
    """One document. pieces (a list of texts) makes it one document whose chunks are exactly those pieces: they are joined with the profile's split_separator and the profile's split_body
    tells the server to split there (a piece over the server's chunk size is still windowed by it). A profile without split_body cannot keep the boundaries, and says so instead of inserting."""
    ep = _lightrag_profile(conn).get("endpoints", {}).get("insert_text", {})
    if pieces is not None and not ep.get("split_body"): return {"error": "this LightRAG profile has no insert_text.split_body - the server version cannot be told where chunks end; use one document per piece instead"}
    sep = ep.get("split_separator", "\n␞\n")
    return await _lightrag_call(conn, "insert_text", {"text": sep.join(pieces) if pieces is not None else text, "source": source or "manual entry"}, extra_body=_render_template(ep["split_body"], {"separator": sep}) if pieces is not None else None)

async def lightrag_insert_texts(conn, texts: list, sources) -> dict:
    """One document per text. sources: one name per text (current LightRAG refuses a batch whose names repeat, and compares basenames), or one name that lightrag_part_names spreads over the texts."""
    return await _lightrag_call(conn, "insert_texts", {"texts": texts, "sources": sources if isinstance(sources, list) else lightrag_part_names(sources or "manual entry", texts)})

def lightrag_part_names(name: str, texts: list) -> list: return [f"{name}__part{k + 1:03d}-{len(texts):03d}" for k in range(len(texts))] if len(texts) > 1 else [name]   # "__" as in lightrag_upload_name: a source reads as a folder of its parts
async def lightrag_insert_file(conn, filename: str, content: bytes) -> dict: return await _lightrag_call(conn, "insert_file", file_bytes=content, file_name=filename)
async def lightrag_list_documents(conn) -> dict: return await _lightrag_call(conn, "list_documents")
async def lightrag_delete_document(conn, doc_id: str, delete_file: bool = True) -> dict: return await lightrag_delete_documents(conn, [doc_id], delete_file)

async def lightrag_delete_documents(conn, doc_ids: list, delete_file: bool = True) -> dict:
    """Deletes documents and what was extracted from them only (entities and relations other documents share are rebuilt). delete_file also removes the server's own uploaded copy,
    which a server needs gone before a changed file of the same name can be uploaded again. Servers refuse while their pipeline is busy: the answer then carries status "busy"."""
    return await _lightrag_call(conn, "delete_document", {"doc_ids": list(doc_ids), "delete_file": delete_file})

def _flatten_docs(r) -> list:
    """The unpaginated document list in either shape seen across versions - {"statuses": {status: [doc, ...]}} or a bare list - as one list, each doc carrying its status."""
    if isinstance(r, list): return r
    groups = r.get("statuses") or r.get("documents") or {}
    if isinstance(groups, list): return groups
    return [{**d, "status": d.get("status") or st} for st, ds in groups.items() for d in (ds if isinstance(ds, list) else [ds]) if isinstance(d, dict)]

async def lightrag_documents(conn, page: int = 1, page_size: int = 100, status: str = "", sort_field: str = "file_path", sort_direction: str = "asc") -> dict:
    """One page of the knowledge server's documents: {documents, pagination: {page, page_size, total_count, total_pages, has_next, has_prev}, status_counts}.
    Servers without the paginated route get their whole list paged here, so callers see one shape."""
    r = await _lightrag_call(conn, "list_documents_paginated", {"page": page, "page_size": page_size, "sort_field": sort_field, "sort_direction": sort_direction, **({"status_filter": status} if status else {})})
    if not _lr_failed(r) or not re.match(r"HTTP 40[45]", r["error"]): return r
    full = await lightrag_list_documents(conn)
    if _lr_failed(full): return full
    docs = _flatten_docs(full)
    counts = {st: sum(1 for d in docs if str(d.get("status", "")).lower() == st) for st in sorted({str(d.get("status", "")).lower() for d in docs})}
    docs = sorted((d for d in docs if not status or str(d.get("status", "")).lower() == status.lower()), key=lambda d: str(d.get(sort_field, "")), reverse=sort_direction == "desc")
    pages = max(1, -(-len(docs) // page_size))
    return {"documents": docs[(page - 1) * page_size:page * page_size], "pagination": {"page": page, "page_size": page_size, "total_count": len(docs), "total_pages": pages, "has_next": page < pages, "has_prev": page > 1}, "status_counts": counts}

def lightrag_upload_name(rel: str) -> str: return rel.replace("\\", "/").strip("/").replace("/", "__")   # the folder path kept in the name ("notes/a.md" -> "notes__a.md"): current LightRAG refuses path separators in uploaded names, earlier versions stripped them
def _lr_name_forms(name: str) -> set: return {name, name.replace("__", "/"), name.replace("__", "")}   # the forms earlier ingests may be stored under: the raw path, or the path with its separators stripped

async def lightrag_doc_index(conn) -> dict:
    """file path -> the server's documents for it, over every page - built once per batch so replacing many files does not list the server per file."""
    out, page = {}, 1
    while True:
        r = await lightrag_documents(conn, page=page, page_size=200)
        if _lr_failed(r): raise RuntimeError(f"""listing documents failed: {r["error"]}""")
        for d in r.get("documents", []): out.setdefault(d.get("file_path", ""), []).append(d)
        if not r.get("pagination", {}).get("has_next"): return out
        page += 1

def _lr_stored(index: dict, name: str, parts: bool = False) -> list:
    """The server's documents stored under name in any form earlier ingests used - with parts, also its <name>__partNNN-NNN pieces."""
    forms = _lr_name_forms(name)
    return [d for k, ds in index.items() if k in forms or (parts and any(k.startswith(f"{f}__part") for f in forms)) for d in ds]

async def _lr_delete_stored(conn, docs: list, wait_s: int) -> dict:
    """Deletes docs (from _lr_stored). A busy pipeline refuses deletion; this waits for it (checking every 15 s) up to wait_s, then gives up."""
    ids, deadline = [d["id"] for d in docs if d.get("id")], time.time() + wait_s
    while ids:
        r = await lightrag_delete_documents(conn, ids)
        if _lr_failed(r): return r
        if r.get("status") != "busy": break
        if time.time() > deadline: return {"error": f"the knowledge server stayed busy for {wait_s} s - earlier version not deleted, nothing inserted"}
        await asyncio.sleep(15)
    return {"deleted": ids}

async def lightrag_replace_file(conn, filename: str, content: bytes, index: dict = None, wait_s: int = 1800) -> dict:
    """Inserts a file (filename = lightrag_upload_name(...)), first deleting the documents the server already holds under that name - so a changed file replaces its earlier version
    instead of being refused as a duplicate (current LightRAG answers HTTP 409) or kept beside it."""
    d = await _lr_delete_stored(conn, _lr_stored(await lightrag_doc_index(conn) if index is None else index, filename), wait_s)
    if _lr_failed(d): return d
    return {**await lightrag_insert_file(conn, filename, content), "replaced": d["deleted"]}

# Hashes of what was inserted under each name, per connection - lets "replace_changed" tell an unchanged source from an edited one without reading the server's copy back.
_MANIFEST_DIR = Path("./data/ai_manager/_knowledge_manifest")
def _manifest(conn) -> dict: return json.loads(p.read_text()) if (p := _MANIFEST_DIR / f"""{conn.get("_id", "conn")}.json""").exists() else {}
def _manifest_set(conn, name: str, digest: str):
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    m = {**_manifest(conn), name: digest}
    (_MANIFEST_DIR / f"""{conn.get("_id", "conn")}.json""").write_text(json.dumps(m, indent=1, sort_keys=True))

ON_EXISTS = ("skip", "replace_changed", "replace", "fail")

async def lightrag_insert_named(conn, name: str, text: str = "", pieces: list = None, separate: bool = False, on_exists: str = "skip", wait_s: int = 1800) -> dict:
    """Inserts text (or pieces) under name, deciding first what to do when the server already holds that name:
    skip - keep what is stored, insert nothing (stored-as-failed documents are reported, not removed: the server's reprocess-failed retries them);
    replace_changed - replace only when the content differs from what this portal last inserted under the name (a name it never inserted counts as changed);
    replace - delete what is stored, then insert; fail - insert anyway, so the server refuses the duplicate.
    pieces with separate=False: one document split at the pieces; separate=True: one document per piece (<name>__partNNN-NNN).
    Answers {"status": "inserted" | "replaced" | "skipped", "name", "stored": [statuses], "deleted": [ids], "response": server answer}."""
    if on_exists not in ON_EXISTS: raise ValueError(f"on_exists must be one of {ON_EXISTS}, got {on_exists!r}")
    digest = hashlib.sha256("\x00".join(pieces if pieces is not None else [text]).encode()).hexdigest()[:16]
    old = _lr_stored(await lightrag_doc_index(conn), name, parts=True) if on_exists != "fail" else []
    stored = sorted({str(d.get("status", "")) for d in old})
    if old and (on_exists == "skip" or (on_exists == "replace_changed" and _manifest(conn).get(name) == digest)): return {"status": "skipped", "name": name, "stored": stored, "deleted": [], "response": {}}
    d = await _lr_delete_stored(conn, old, wait_s) if old else {"deleted": []}
    if _lr_failed(d): return d
    r = await (lightrag_insert_texts(conn, pieces, lightrag_part_names(name, pieces)) if pieces is not None and separate else lightrag_insert_text(conn, text, name, pieces=pieces))
    if _lr_failed(r): return r
    _manifest_set(conn, name, digest)
    return {"status": "replaced" if d["deleted"] else "inserted", "name": name, "stored": stored, "deleted": d["deleted"], "response": r}

async def lightrag_clear_all(conn) -> dict:
    r = await _lightrag_call(conn, "clear_all")
    return {"ok": True} if "error" not in r else r

async def lightrag_query_cached(conn, query: str, mode: str = "hybrid", ttl: int = 300, **extra) -> dict:
    key = (conn.get("_id", id(conn)), mode, query, tuple(sorted(extra.items())), tuple(sorted((conn.get("query_defaults") or {}).items())))
    now = time.time()
    hit = _QUERY_CACHE.get(key)
    if hit and now - hit[0] < ttl: return hit[1]
    result = await lightrag_query(conn, query, mode, **extra)
    _QUERY_CACHE[key] = (now, result)
    return result

async def lightrag_query_data(conn, query: str, mode: str = "mix", **extra) -> dict:
    """Retrieval only - the entities, relationships and text chunks a query finds, with no answer written by the knowledge server, so any chat model can answer from them.
    Returns {"context": text block, "data": structured result}; servers without /query/data fall back to a query with only_need_context."""
    opts = {k: extra.pop(k, v) for k, v in lightrag_query_defaults(conn).items()}
    r = await _lightrag_call(conn, "query_data", {"query": query, "mode": mode}, extra_body={**{k: v for k, v in opts.items() if v is not None and k != "only_need_context"}, **{k: v for k, v in extra.items() if v is not None}})
    if _lr_failed(r) and re.match(r"HTTP 40[45]", r["error"]):
        r = await lightrag_query(conn, query, mode, **{**extra, "only_need_context": True})
        return r if _lr_failed(r) else {"context": r.get("response", ""), "data": {}}
    if _lr_failed(r): return r
    data = r.get("data", r)
    return {"context": _format_query_data(data), "data": data}

def _format_query_data(d: dict) -> str:
    """Structured retrieval as one plain-text context block for a chat model."""
    ents = "\n".join(f"""- {e.get("entity_name") or e.get("entity") or e.get("id", "?")} ({e.get("entity_type", "")}): {e.get("description", "")}""" for e in d.get("entities", []))
    rels = "\n".join(f"""- {r.get("src_id") or r.get("source", "?")} -> {r.get("tgt_id") or r.get("target", "?")}: {r.get("description", "")}""" for r in d.get("relationships", []))
    chunks = "\n\n".join(f"""[{c.get("file_path", "")}]\n{c.get("content", "")}""" for c in d.get("chunks", []))
    return "\n\n".join(part for part in (f"Entities:\n{ents}" if ents else "", f"Relationships:\n{rels}" if rels else "", f"Text:\n{chunks}" if chunks else "") if part)

async def lightrag_context_block(conn, query: str, mode: str = "hybrid", label: str = "Knowledge") -> str:
    r = await lightrag_query_cached(conn, query, mode)
    text = r.get("response", "").strip()
    return f"[{label}]\n{text}\n" if text else ""

def _parse_graph_response(data) -> tuple:
    """Normalizes the two response shapes seen across LightRAG versions/forks - {"nodes":[...],"edges":[...]} or a bare list of label strings/objects. Extend here (the one shared place), never per-caller, if a new shape appears."""
    if isinstance(data, dict): return data.get("nodes", []), data.get("edges", [])
    if isinstance(data, list): return [{"id": i, "label": i} if isinstance(i, str) else i for i in data], []
    return [], []

async def lightrag_graph(conn, label: str = "*", max_depth: int = 3, max_nodes: int = 1000) -> dict:
    """The subgraph around a label ("*" = the whole graph) as {"nodes", "edges", "is_truncated"} - the server cuts it at max_nodes."""
    data = await _lightrag_call(conn, "graph", {"limit": max_nodes, "label": label, "max_depth": max_depth, "max_nodes": max_nodes})
    if _lr_failed(data): return data
    nodes, edges = _parse_graph_response(data)
    return {"nodes": nodes, "edges": edges, "is_truncated": bool(data.get("is_truncated")) if isinstance(data, dict) else False}

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
    """/system/load switches the node to model_path (it unloads what it holds first, and keeps it when model_path does not exist); while a generation runs it answers at once with status Error."""
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