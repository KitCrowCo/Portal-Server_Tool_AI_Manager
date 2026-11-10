"""conntections.py
Shared utilities for all ai tools conenctions.
Not a sub-module (no subdirectory/router) - import directly.
"""
import json, httpx, time, re, asyncio, hashlib, uuid, contextvars
from pathlib import Path

_QUERY_CACHE: dict = {}
_MISSING = object()
_TEMPLATE_CACHE: dict = {}
KG_DIR = Path("./data/ai_tools/_knowledge")
COMMON_DIR = Path("./data/_common")
_TMPL_DIR = Path("./tools/ai_manager/_connections")   # provider schema profiles - code-shipped
CONN_DIR  = Path("./data/_tools/ai_manager/_connections")    # saved connection instances - tool-owned data, cross-module
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

def unreachable_hint(conn) -> str:
    """Why "All connection attempts failed" usually happens, said where the error is shown: nothing answered at this address from where the portal itself runs."""
    host = str(conn.get("values", {}).get("host", ""))
    local = " localhost and 127.0.0.1 are the portal's own container, not the machine it runs on: use the machine's address." if host.split("://")[-1].split(":")[0] in ("localhost", "127.0.0.1") else ""
    return f"""Nothing answered at {_base(conn)} from where this portal runs.{local} If another computer reaches that address but this portal does not, the portal's container network or the target machine's firewall keeps it out: a container reaches its own host from Docker's bridge network (172.16.0.0/12), not the LAN, so a firewall rule allowing only the LAN refuses it."""

async def list_models_async(conn) -> list:
    if not conn: return []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)) as c:
            r = await c.get(f"{_base(conn)}/api/tags", headers=_auth_headers(conn))
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
            r = await c.post(f"{_base(conn)}/api/embed", json={"model": model, "input": text}, headers=_auth_headers(conn))
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

_LIVE_CAPS: dict = {}   # server address -> (time asked, capabilities it reports, or None when it did not answer)

def _live_caps(conn: dict, spec: dict):
    """The capability names a node reports of itself, asked once per ttl_s (15 s after a failed try). None when it has never answered: the profile's list is then all that is known, and the call itself shows the failure. The node answers without starting anything."""
    base = _base(conn)
    t, caps = _LIVE_CAPS.get(base, (0.0, None))
    if time.time() - t < (float(spec.get("ttl_s", 60)) if caps is not None else 15.0): return caps
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=2.0, read=3.0, write=2.0, pool=2.0)) as c:
            r = c.get(base + spec["path"], headers=_auth_headers(conn))
            caps = set(r.json().get("capabilities", {})) if r.status_code == 200 else caps
    except Exception: pass   # an unreachable node is an external failure: keep what it last said
    _LIVE_CAPS[base] = (time.time(), caps)
    return caps

def connection_capabilities(conn: dict) -> dict:
    """What this connection can do - checked before ever routing a request to it. A missing/false key always means 'not supported', never assumed true. A profile with "live_capabilities" narrows its list to what the node itself reports: one connection type, whatever modules each node runs."""
    profile = _conn_profile(conn)
    caps, spec = profile.get("capabilities", {}), profile.get("live_capabilities")
    live = _live_caps(conn, spec) if spec else None
    return caps if live is None else {k: v for k, v in caps.items() if k not in spec["subject"] or spec.get("map", {}).get(k, k) in live}

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

def _add_fragments(frags: dict, calls: list):
    """Streamed OpenAI-style tool calls arrive in pieces keyed by index: the first with the id and name, the rest with more of the arguments string. Assembled in frags, yielded once when the stream ends."""
    for tc in calls:
        fn, i = tc.get("function") or {}, tc.get("index", len(frags) if tc.get("id") or not frags else max(frags))   # no index: a piece with an id starts a call, one without continues the last
        s, a = frags.setdefault(i, {"id": "", "function": {"name": "", "arguments": ""}}), fn.get("arguments")
        n = s["function"]["name"]
        s["id"], s["function"]["name"], s["function"]["arguments"] = s["id"] or tc.get("id") or "", n if (fn.get("name") or "") in ("", n) else n + fn["name"], s["function"]["arguments"] + (a if isinstance(a, str) else json.dumps(a) if a else "")   # a server repeating the name on every piece is not doubled

# --- Request ids and the control channel ---
# A profile with "request_id_header" gets an id on every chat call (<job id>.<random>, from CALL_TAG - the engine sets it for each node it runs). The calls in flight are kept per job, so cancel_calls(job) can ask each server to drop them (capability request_cancel).
# A profile with "liveness": {"capability", "after_s"} is asked, whenever a stream has been silent for after_s, whether it is still working on the request (capability node_status answers {"requests": [{"id"}...]}); a server that no longer knows the request ends the call at once instead of after the read timeout.

CALL_TAG = contextvars.ContextVar("ai_manager_call_tag", default="")
_CALLS: dict = {}   # tag (job id) -> {request id: connection}

def calls_in_flight(tag: str) -> dict: return dict(_CALLS.get(tag, {}))

async def cancel_calls(tag: str) -> int:
    """Asks the server of every call in flight for tag to drop it; returns how many servers said they did. Servers without request_cancel are skipped (closing the connection is all they get)."""
    n = 0
    for rid, conn in calls_in_flight(tag).items():
        if has_capability(conn, "request_cancel"): n += bool((await call_capability(conn, "request_cancel", {"request_id": rid}, timeout_s=5)).get("cancelled"))
    return n

async def _probed_lines(resp, conn: dict, rid: str):
    """resp.aiter_lines(), with the profile's liveness check while the stream is silent: the pending read is never cancelled, the server is asked about rid every after_s. Three unanswered checks in a row, or an answer without rid, end the call."""
    live, it = _conn_profile(conn).get("liveness") or {}, resp.aiter_lines().__aiter__()
    if not (rid and live.get("capability") and has_capability(conn, live["capability"])):
        async for line in it: yield line
        return
    after, misses, nxt = float(live.get("after_s", 60)), 0, None
    try:
        while True:
            nxt = nxt or asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({nxt}, timeout=after)
            if not done:
                st = await call_capability(conn, live["capability"], {}, timeout_s=10)
                if str(st.get("error", "")).startswith("HTTP 404"): after = None; continue   # a server older than its profile (no status endpoint): wait as without the check
                misses = misses + 1 if st.get("error") else 0
                if misses >= 3: raise RuntimeError(f"""{conn.get("_id", "?")} stopped answering its status check ({st["error"]}) while this request was silent - treated as lost""")
                if not st.get("error") and rid not in {r.get("id") for r in st.get("requests", [])}: raise RuntimeError(f"""{conn.get("_id", "?")} is no longer working on this request (it restarted, or the request was dropped)""")
                continue
            try: line = nxt.result()
            except StopAsyncIteration: return
            nxt = None
            yield line
    finally:
        if nxt and not nxt.done(): nxt.cancel()

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
    timeout_s, tag, rid_header = float(conn.get("values", {}).get("timeout_s", 3000)), CALL_TAG.get(), profile.get("request_id_header")
    rid = f"""{tag or "call"}.{uuid.uuid4().hex[:8]}""" if rid_header else ""
    frags = {}   # sse tool-call pieces by index (_add_fragments)
    if rid and tag: _CALLS.setdefault(tag, {})[rid] = conn
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=10.0, pool=10.0)) as c:
            async with c.stream(chat_ep.get("method","POST"), url, json=payload, headers={**_auth_headers(conn), **({rid_header: rid} if rid else {})}) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(f"""connection error {resp.status_code} calling {conn.get('_id','?')} model '{model}': {body.decode(errors='replace')[:500]}""")
                async for line in _probed_lines(resp, conn, rid):
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
                    if isinstance(calls, list) and parser.get("stream_format") == "sse": _add_fragments(frags, calls); calls = []
                    if text or thinking or calls: yield {"text": text or "", "thinking": thinking or "", "tool_calls": [_norm_tool_call(tc) for tc in calls] if isinstance(calls, list) else []}
                    if done_sentinel and chunk.get(done_sentinel.get("key","done")) == done_sentinel.get("value", True): break
                if frags: yield {"text": "", "thinking": "", "tool_calls": [_norm_tool_call(frags[i]) for i in sorted(frags)]}
    finally:
        if rid and tag and _CALLS.get(tag, {}).pop(rid, None) and not _CALLS[tag]: _CALLS.pop(tag)

async def stream_llm(conn: dict, messages: list, model: str, think = False, **kwargs):
    """(text, thinking) tuple view over stream_llm_events, for every caller that doesn't use tool calls."""
    async for ev in stream_llm_events(conn, messages, model, think, **kwargs):
        if ev["text"] or ev["thinking"]: yield ev["text"], ev["thinking"]

def has_capability(conn: dict, capability: str, model: str = "") -> bool:
    """True/False capabilities are static. A dict capability checks its 'models' sublayer -
    absent model or empty sublayer defaults to capable, so declaring the sublayer costs nothing until you actually start populating per-model exceptions."""
    cap = connection_capabilities(conn).get(capability)
    if isinstance(cap, dict):
        models = cap.get("models", {})
        return models.get(model, True) if model else True
    return bool(cap)

def _capability_request(conn: dict, capability: str, payload: dict) -> tuple:
    """(method, url, body) for one capability from the connection's profile, or (None, None, error text) - shared by call_capability and call_capability_sync."""
    if not has_capability(conn, capability): return None, None, f"""connection '{conn.get('_id','?')}' ({conn.get('connection_type','?')}) does not declare capability '{capability}'"""
    ep = _conn_profile(conn).get("endpoints", {}).get(capability, {})
    if not ep: return None, None, f"""profile declares '{capability}' but has no endpoints.{capability} - profile misconfigured"""
    return ep.get("method", "POST"), _base(conn) + _fmt_path(ep.get("path", "/"), payload), payload if ep.get("body_all") else _render_template(ep.get("body", {}), payload)

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

# --- Knowledge groups (config-driven: a connection profile's "knowledge" block) ---
# A knowledge group is what every knowledge_* operation works on: a connection-shaped dict whose "_id" is the group's reference.
#   A server holding one group (a LightRAG server's own connection): the connection itself, reference = its id.
#   A node holding many (Magpie): reference "<connection id>/<group>", the group in "_group"; the profile's knowledge.base_path ("/knowledge/{group}") leads to it.
# The wire format (paths, bodies, answers) is the profile its connection names in knowledge.endpoints_from - a JSON edit for another server or version, never a code change here.

_GROUPS: dict = {}   # connection id -> (time read, [group dicts]) for nodes that list their groups
_GROUP_META = Path("./data/_tools/ai_manager/_knowledge_groups.json")   # per group reference: settings kept on this portal (query defaults) for groups that are not connections of their own

def _kspec(conn: dict): return _conn_profile(conn).get("knowledge")   # None: this connection offers no knowledge
def knowledge_profile(conn: dict) -> dict: return json.loads((_TMPL_DIR / f"""{_kspec(conn).get("endpoints_from") or conn.get("connection_type")}.json""").read_text())
def _kbase(conn: dict) -> str: return _base(conn) + _fmt_path(_kspec(conn).get("base_path", ""), {"group": conn.get("_group", "")})
def _group_meta() -> dict: return json.loads(_GROUP_META.read_text()) if _GROUP_META.exists() else {}

def _group_target(conn: dict, g: dict) -> dict:
    ref = f"""{conn["_id"]}/{g["name"]}"""
    return {**conn, "_id": ref, "_conn_id": conn["_id"], "_group": g["name"], "display_name": f"""{g["name"]} ({conn.get("display_name", conn["_id"])})""", "values": {**conn.get("values", {}), "domain_notes": g.get("notes", "")}, "group": g, "query_defaults": _group_meta().get(ref, {}).get("query_defaults") or {}}

def _groups_of(conn: dict) -> list:
    """The knowledge groups one connection offers: itself for a single-group server; a node's own list (read with short timeouts, kept 30 s - an unreachable node offers none until it answers)."""
    ks = _kspec(conn)
    if ks is None: return []
    if not ks.get("groups_path"): return [conn]
    hit = _GROUPS.get(conn["_id"])
    if not hit or time.time() - hit[0] > 30:
        try:
            r = httpx.get(_base(conn) + ks["groups_path"], headers=_auth_headers(conn), timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0))
            found = r.json().get("groups", []) if r.status_code == 200 else []
        except (httpx.HTTPError, ValueError): found = []
        hit = _GROUPS[conn["_id"]] = (time.time(), found)
    return [_group_target(conn, g) for g in hit[1]]

def forget_groups(conn_id: str = ""): _GROUPS.pop(conn_id, None) if conn_id else _GROUPS.clear()   # after groups were added or removed on a node

def knowledge_groups() -> list:
    """Every knowledge group this portal reaches, from every connection that offers knowledge."""
    return [t for c in list_conns(get_all=True) for t in _groups_of(c)]

def knowledge_target(ref: str):
    """A knowledge group by its reference (see above), or None when its connection is gone. A node's group is not checked here: an operation on a missing group answers the node's error."""
    if not ref: return None
    cid, _, g = ref.partition("/")
    c = get_conn(cid)
    if not c or _kspec(c) is None: return None
    if not g: return c if not _kspec(c).get("groups_path") else None
    return next((t for t in _groups_of(c) if t["_group"] == g), None) or _group_target(c, {"name": g})

def knowledge_save_defaults(target: dict, defaults: dict):
    """A group's own query defaults: kept in its connection when it is one, else on this portal by reference."""
    if not target.get("_group"):
        raw = load_conn_raw(target["_id"])
        raw["query_defaults"] = defaults
        return save_conn(target["_id"], raw)
    _GROUP_META.parent.mkdir(parents=True, exist_ok=True)
    meta = _group_meta()
    meta.setdefault(target["_id"], {})["query_defaults"] = defaults
    _GROUP_META.write_text(json.dumps(meta, indent=1))

def _fmt_path(path: str, params: dict) -> str:
    for k, v in params.items(): path = path.replace("{" + str(k) + "}", str(v))
    return path

async def _knowledge_call(conn, endpoint_name: str, params: dict = None, extra_body: dict = None, file_bytes: bytes = None, file_name: str = "") -> dict:
    profile = knowledge_profile(conn)
    ep = profile.get("endpoints", {}).get(endpoint_name, {})
    if not ep: return {"error": f"""the knowledge profile has no {endpoint_name} endpoint declared - add it to its JSON profile"""}
    params = params or {}
    method, path = ep.get("method", "GET"), _fmt_path(ep.get("path", "/"), params)
    fallback = _fmt_path(ep.get("fallback_path", ""), params) if ep.get("fallback_path") else ""
    body = _render_template(ep.get("body", {}), params) if ep.get("body") else None
    if body is not None and extra_body: body = {**body, **extra_body}
    read_timeout = float(conn.get("values", {}).get("timeout_s", 2400))
    async def _do(c, p):
        if file_bytes is not None: return await c.request(method, _kbase(conn)+p, files={ep.get("file_field","file"): (file_name, file_bytes)}, headers=_auth_headers(conn))
        if method == "GET": return await c.request(method, _kbase(conn)+p, params=body, headers=_auth_headers(conn))
        return await c.request(method, _kbase(conn)+p, json=body, headers=_auth_headers(conn))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=read_timeout, write=60.0, pool=20.0)) as c:
            r = await _do(c, path)
            if r.status_code == 404 and fallback: r = await _do(c, fallback)
            if r.status_code not in (200, 204): return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
            return r.json() if r.content else {"ok": True}
    except Exception as e: return {"error": str(e)}

def knowledge_query_options_schema(conn) -> dict:
    """UI callers (Kimi) render this dynamically as query-time toggles/fields - a new option is a JSON edit, never a new checkbox hand-added to kimi.py."""
    return knowledge_profile(conn).get("query_options_schema", {})

def knowledge_query_defaults(conn) -> dict:
    """Query options in effect for one knowledge group: the profile's schema defaults, overlaid with the group's own saved defaults (conn["query_defaults"], set from Kimi).
    Every caller (Kimi, knowledge nodes, context blocks) gets the same per-group behaviour, e.g. rerank on for one group and off for another."""
    return {**{k: spec.get("default") for k, spec in knowledge_query_options_schema(conn).items()}, **(conn.get("query_defaults") or {})}

def base_url(conn) -> str: return _base(conn)   # for links to a server's own pages (e.g. LightRAG's WebUI)

async def knowledge_op(conn, name: str, **params) -> dict:
    """Any operation declared under endpoints in the connection's profile, by name - pipeline_status, track_status (track_id=...), reprocess_failed, cancel_pipeline, status_counts, graph_labels, openapi.
    A server without that route answers with its own HTTP error, returned as {"error": ...}."""
    return await _knowledge_call(conn, name, params)

async def knowledge_health(conn) -> dict:
    r = await _knowledge_call(conn, "health")
    return {"ok": "error" not in r, "detail": r.get("error", r)}

async def knowledge_query_stream(conn, query: str, mode: str = "mix", **extra):
    """NDJSON streaming query against /query/stream per LightRAG's documented contract.
    Yields (chunk_text, references, error) - references arrive on the first line if requested, error terminates iteration."""
    profile = knowledge_profile(conn)
    ep = profile.get("endpoints", {}).get("query_stream", {})
    if not ep: yield "", None, "the knowledge profile has no query_stream endpoint declared"; return
    opts = {k: extra.pop(k, v) for k, v in knowledge_query_defaults(conn).items()}
    body = {"query": query, "mode": mode, "stream": True, **{k:v for k,v in opts.items() if v is not None}, **{k:v for k,v in extra.items() if v is not None}}
    read_timeout = float(conn.get("values", {}).get("timeout_s", 2400))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=20.0, read=read_timeout, write=60.0, pool=20.0)) as c:
            async with c.stream("POST", _kbase(conn) + ep.get("path", "/query/stream"), json=body, headers=_auth_headers(conn)) as resp:
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

async def knowledge_query(conn, query: str, mode: str = "hybrid", **extra) -> dict:
    opts = {k: extra.pop(k, v) for k, v in knowledge_query_defaults(conn).items()}
    extra_body = {**{k:v for k,v in opts.items() if v is not None}, **{k:v for k,v in extra.items() if v is not None}}
    r = await _knowledge_call(conn, "query", {"query": query, "mode": mode}, extra_body=extra_body)
    if _kn_failed(r): return r
    parser = knowledge_profile(conn).get("endpoints",{}).get("query",{}).get("response_parser", {"content_path":"response"})
    return {"response": _get_nested_value(r, parser.get("content_path","response")) or r.get("response",""), "raw": r}

async def knowledge_insert_text(conn, text: str, source: str = "", pieces: list = None) -> dict:
    """One document. pieces (a list of texts) makes it one document whose chunks are exactly those pieces: they are joined with the profile's split_separator and the profile's split_body
    tells the server to split there (a piece over the server's chunk size is still windowed by it). A profile without split_body cannot keep the boundaries, and says so instead of inserting."""
    ep = knowledge_profile(conn).get("endpoints", {}).get("insert_text", {})
    if pieces is not None and not ep.get("split_body"): return {"error": "this knowledge profile has no insert_text.split_body - the server version cannot be told where chunks end; use one document per piece instead"}
    sep = ep.get("split_separator", "\n␞\n")
    return await _knowledge_call(conn, "insert_text", {"text": sep.join(pieces) if pieces is not None else text, "source": source or "manual entry"}, extra_body=_render_template(ep["split_body"], {"separator": sep}) if pieces is not None else None)

async def knowledge_insert_texts(conn, texts: list, sources) -> dict:
    """One document per text. sources: one name per text (current LightRAG refuses a batch whose names repeat, and compares basenames), or one name that knowledge_part_names spreads over the texts."""
    return await _knowledge_call(conn, "insert_texts", {"texts": texts, "sources": sources if isinstance(sources, list) else knowledge_part_names(sources or "manual entry", texts)})

def knowledge_part_names(name: str, texts: list) -> list: return [f"{name}__part{k + 1:03d}-{len(texts):03d}" for k in range(len(texts))] if len(texts) > 1 else [name]   # "__" as in knowledge_upload_name: a source reads as a folder of its parts
async def knowledge_insert_file(conn, filename: str, content: bytes) -> dict: return await _knowledge_call(conn, "insert_file", file_bytes=content, file_name=filename)
async def knowledge_list_documents(conn) -> dict: return await _knowledge_call(conn, "list_documents")
async def knowledge_delete_document(conn, doc_id: str, delete_file: bool = True) -> dict: return await knowledge_delete_documents(conn, [doc_id], delete_file)

async def knowledge_delete_documents(conn, doc_ids: list, delete_file: bool = True) -> dict:
    """Deletes documents and what was extracted from them only (entities and relations other documents share are rebuilt).
    Delete_file also removes the server's own uploaded copy, which a server needs gone before a changed file of the same name can be uploaded again.
    Servers refuse while their pipeline is busy: the answer then carries status "busy"."""
    return await _knowledge_call(conn, "delete_document", {"doc_ids": list(doc_ids), "delete_file": delete_file})

def _flatten_docs(r) -> list:
    """The unpaginated document list in either shape seen across versions - {"statuses": {status: [doc, ...]}} or a bare list - as one list, each doc carrying its status."""
    if isinstance(r, list): return r
    groups = r.get("statuses") or r.get("documents") or {}
    if isinstance(groups, list): return groups
    return [{**d, "status": d.get("status") or st} for st, ds in groups.items() for d in (ds if isinstance(ds, list) else [ds]) if isinstance(d, dict)]

async def knowledge_documents(conn, page: int = 1, page_size: int = 100, status: str = "", sort_field: str = "file_path", sort_direction: str = "asc") -> dict:
    """One page of the knowledge server's documents: {documents, pagination: {page, page_size, total_count, total_pages, has_next, has_prev}, status_counts}.
    Servers without the paginated route get their whole list paged here, so callers see one shape."""
    r = await _knowledge_call(conn, "list_documents_paginated", {"page": page, "page_size": page_size, "sort_field": sort_field, "sort_direction": sort_direction, **({"status_filter": status} if status else {})})
    if not _kn_failed(r) or not re.match(r"HTTP 40[45]", r["error"]): return r
    full = await knowledge_list_documents(conn)
    if _kn_failed(full): return full
    docs = _flatten_docs(full)
    counts = {st: sum(1 for d in docs if str(d.get("status", "")).lower() == st) for st in sorted({str(d.get("status", "")).lower() for d in docs})}
    docs = sorted((d for d in docs if not status or str(d.get("status", "")).lower() == status.lower()), key=lambda d: str(d.get(sort_field, "")), reverse=sort_direction == "desc")
    pages = max(1, -(-len(docs) // page_size))
    return {"documents": docs[(page - 1) * page_size:page * page_size], "pagination": {"page": page, "page_size": page_size, "total_count": len(docs), "total_pages": pages, "has_next": page < pages, "has_prev": page > 1}, "status_counts": counts}

def knowledge_upload_name(rel: str) -> str: return rel.replace("\\", "/").strip("/").replace("/", "__")   # the folder path kept in the name ("notes/a.md" -> "notes__a.md"): current LightRAG refuses path separators in uploaded names, earlier versions stripped them
def _kn_name_forms(name: str) -> set: return {name, name.replace("__", "/"), name.replace("__", "")}   # the forms earlier ingests may be stored under: the raw path, or the path with its separators stripped

async def knowledge_doc_index(conn) -> dict:
    """file path -> the server's documents for it, over every page - built once per batch so replacing many files does not list the server per file."""
    out, page = {}, 1
    while True:
        r = await knowledge_documents(conn, page=page, page_size=200)
        if _kn_failed(r): raise RuntimeError(f"""listing documents failed: {r["error"]}""")
        for d in r.get("documents", []): out.setdefault(d.get("file_path", ""), []).append(d)
        if not r.get("pagination", {}).get("has_next"): return out
        page += 1

def _kn_stored(index: dict, name: str, parts: bool = False) -> list:
    """The server's documents stored under name in any form earlier ingests used - with parts, also its <name>__partNNN-NNN pieces."""
    forms = _kn_name_forms(name)
    return [d for k, ds in index.items() if k in forms or (parts and any(k.startswith(f"{f}__part") for f in forms)) for d in ds]

async def _kn_delete_stored(conn, docs: list, wait_s: int) -> dict:
    """Deletes docs (from _kn_stored). A busy pipeline refuses deletion; this waits for it (checking every 15 s) up to wait_s, then gives up."""
    ids, deadline = [d["id"] for d in docs if d.get("id")], time.time() + wait_s
    while ids:
        r = await knowledge_delete_documents(conn, ids)
        if _kn_failed(r): return r
        if r.get("status") != "busy": break
        if time.time() > deadline: return {"error": f"the knowledge server stayed busy for {wait_s} s - earlier version not deleted, nothing inserted"}
        await asyncio.sleep(15)
    return {"deleted": ids}

async def knowledge_replace_file(conn, filename: str, content: bytes, index: dict = None, wait_s: int = 1800) -> dict:
    """Inserts a file (filename = knowledge_upload_name(...)), first deleting the documents the server already holds under that name - so a changed file replaces its earlier version instead of being refused as a duplicate (current LightRAG answers HTTP 409) or kept beside it."""
    d = await _kn_delete_stored(conn, _kn_stored(await knowledge_doc_index(conn) if index is None else index, filename), wait_s)
    if _kn_failed(d): return d
    return {**await knowledge_insert_file(conn, filename, content), "replaced": d["deleted"]}

async def knowledge_delete_file(conn, filename: str, index: dict = None, wait_s: int = 1800) -> dict:
    """Deletes every document the server holds under filename (knowledge_upload_name(...)), in any form earlier ingests stored it - for a source file that no longer exists. {"deleted": [ids]} or {"error"}."""
    return await _kn_delete_stored(conn, _kn_stored(await knowledge_doc_index(conn) if index is None else index, filename), wait_s)

async def knowledge_delete_files(conn, filenames: list, index: dict = None, wait_s: int = 1800) -> dict:
    """knowledge_delete_file for many names in one request: the server works through it as one background job, instead of a busy wait between every two files. {"deleted": [ids]} or {"error"}."""
    index = await knowledge_doc_index(conn) if index is None else index
    return await _kn_delete_stored(conn, [d for n in filenames for d in _kn_stored(index, n)], wait_s)

# Hashes of what was inserted under each name, per connection - lets "replace_changed" tell an unchanged source from an edited one without reading the server's copy back.
_MANIFEST_DIR = Path("./data/_tools/ai_manager/_knowledge_manifest")
def _manifest_path(conn) -> Path: return _MANIFEST_DIR / f"""{conn.get("_id", "conn").replace("/", "__")}.json"""   # a node's group reference holds a "/"
def _manifest(conn) -> dict: return json.loads(p.read_text()) if (p := _manifest_path(conn)).exists() else {}
def _manifest_set(conn, name: str, digest: str):
    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    m = {**_manifest(conn), name: digest}
    _manifest_path(conn).write_text(json.dumps(m, indent=1, sort_keys=True))

ON_EXISTS = ("skip", "replace_changed", "replace", "fail")

async def knowledge_insert_named(conn, name: str, text: str = "", pieces: list = None, separate: bool = False, on_exists: str = "skip", wait_s: int = 1800, index: dict = None) -> dict:
    """Inserts text (or pieces) under name, deciding first what to do when the server already holds that name:
    skip - keep what is stored, insert nothing (stored-as-failed documents are reported, not removed: the server's reprocess-failed retries them);
    replace_changed - replace only when the content differs from what this portal last inserted under the name (a name it never inserted counts as changed);
    replace - delete what is stored, then insert; fail - insert anyway, so the server refuses the duplicate.
    pieces with separate=False: one document split at the pieces; separate=True: one document per piece (<name>__partNNN-NNN).
    index: a knowledge_doc_index built once for a batch, so a batch of names does not list the server per name.
    Answers {"status": "inserted" | "replaced" | "skipped", "name", "stored": [statuses], "deleted": [ids], "response": server answer}."""
    if on_exists not in ON_EXISTS: raise ValueError(f"on_exists must be one of {ON_EXISTS}, got {on_exists!r}")
    digest = hashlib.sha256("\x00".join(pieces if pieces is not None else [text]).encode()).hexdigest()[:16]
    old = _kn_stored(await knowledge_doc_index(conn) if index is None else index, name, parts=True) if on_exists != "fail" else []
    stored = sorted({str(d.get("status", "")) for d in old})
    done = [d for d in old if str(d.get("status", "")).lower() == "processed"]   # only a successfully-processed doc counts as "already there": a failed or still-pending one is retried, so a re-run after a crash picks up what did not finish
    if done and (on_exists == "skip" or (on_exists == "replace_changed" and _manifest(conn).get(name) == digest)): return {"status": "skipped", "name": name, "stored": stored, "deleted": [], "response": {}}
    d = await _kn_delete_stored(conn, old, wait_s) if old else {"deleted": []}
    if _kn_failed(d): return d
    r = await (knowledge_insert_texts(conn, pieces, knowledge_part_names(name, pieces)) if pieces is not None and separate else knowledge_insert_text(conn, text, name, pieces=pieces))
    if _kn_failed(r): return r
    _manifest_set(conn, name, digest)
    return {"status": "replaced" if d["deleted"] else "inserted", "name": name, "stored": stored, "deleted": d["deleted"], "response": r}

async def knowledge_clear_all(conn) -> dict:
    r = await _knowledge_call(conn, "clear_all")
    return {"ok": True} if "error" not in r else r

async def knowledge_query_cached(conn, query: str, mode: str = "hybrid", ttl: int = 300, **extra) -> dict:
    key = (conn.get("_id", id(conn)), mode, query, tuple(sorted(extra.items())), tuple(sorted((conn.get("query_defaults") or {}).items())))
    now = time.time()
    hit = _QUERY_CACHE.get(key)
    if hit and now - hit[0] < ttl: return hit[1]
    result = await knowledge_query(conn, query, mode, **extra)
    _QUERY_CACHE[key] = (now, result)
    return result

async def knowledge_query_data(conn, query: str, mode: str = "mix", **extra) -> dict:
    """Retrieval only - the entities, relationships and text chunks a query finds, with no answer written by the knowledge server, so any chat model can answer from them.
    Returns {"context": text block, "data": structured result}; servers without /query/data fall back to a query with only_need_context."""
    opts = {k: extra.pop(k, v) for k, v in knowledge_query_defaults(conn).items()}
    r = await _knowledge_call(conn, "query_data", {"query": query, "mode": mode}, extra_body={**{k: v for k, v in opts.items() if v is not None and k != "only_need_context"}, **{k: v for k, v in extra.items() if v is not None}})
    if _kn_failed(r) and re.match(r"HTTP 40[45]", r["error"]):
        r = await knowledge_query(conn, query, mode, **{**extra, "only_need_context": True})
        return r if _kn_failed(r) else {"context": r.get("response", ""), "data": {}}
    if _kn_failed(r): return r
    data = r.get("data", r)
    return {"context": _format_query_data(data), "data": data}

def _format_query_data(d: dict) -> str:
    """Structured retrieval as one plain-text context block for a chat model."""
    ents = "\n".join(f"""- {e.get("entity_name") or e.get("entity") or e.get("id", "?")} ({e.get("entity_type", "")}): {e.get("description", "")}""" for e in d.get("entities", []))
    rels = "\n".join(f"""- {r.get("src_id") or r.get("source", "?")} -> {r.get("tgt_id") or r.get("target", "?")}: {r.get("description", "")}""" for r in d.get("relationships", []))
    chunks = "\n\n".join(f"""[{c.get("file_path", "")}]\n{c.get("content", "")}""" for c in d.get("chunks", []))
    return "\n\n".join(part for part in (f"Entities:\n{ents}" if ents else "", f"Relationships:\n{rels}" if rels else "", f"Text:\n{chunks}" if chunks else "") if part)

async def knowledge_context_block(conn, query: str, mode: str = "hybrid", label: str = "Knowledge") -> str:
    r = await knowledge_query_cached(conn, query, mode)
    text = r.get("response", "").strip()
    return f"[{label}]\n{text}\n" if text else ""

def _parse_graph_response(data) -> tuple:
    """Normalizes the two response shapes seen across LightRAG versions/forks - {"nodes":[...],"edges":[...]} or a bare list of label strings/objects. Extend here (the one shared place), never per-caller, if a new shape appears."""
    if isinstance(data, dict): return data.get("nodes", []), data.get("edges", [])
    if isinstance(data, list): return [{"id": i, "label": i} if isinstance(i, str) else i for i in data], []
    return [], []

async def knowledge_graph(conn, label: str = "*", max_depth: int = 3, max_nodes: int = 1000) -> dict:
    """The subgraph around a label ("*" = the whole graph) as {"nodes", "edges", "is_truncated"} - the server cuts it at max_nodes."""
    data = await _knowledge_call(conn, "graph", {"limit": max_nodes, "label": label, "max_depth": max_depth, "max_nodes": max_nodes})
    if _kn_failed(data): return data
    nodes, edges = _parse_graph_response(data)
    return {"nodes": nodes, "edges": edges, "is_truncated": bool(data.get("is_truncated")) if isinstance(data, dict) else False}

async def knowledge_graph_dot(conn, limit: int = 1000, label: str = "*") -> str:
    data = await _knowledge_call(conn, "graph", {"limit": limit, "label": label, "max_depth": 3, "max_nodes": limit})
    if _kn_failed(data): print(f"[LightRAG] graph fetch error: {data['error']}"); return ""
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

async def knowledge_list_entities(conn, limit: int = 1000) -> list:
    data = await _knowledge_call(conn, "graph", {"limit": limit})
    if _kn_failed(data): print(f"[LightRAG] graph fetch error: {data['error']}"); return []
    nodes, _ = _parse_graph_response(data)
    return nodes

def _kn_failed(r) -> bool:
    """True only on a REAL failure. Only our own _knowledge_call ever sets 'error' to a truthy message (on an exception or non-2xx status) - a response dict that merely contains a falsy 'error' key (some LightRAG versions include this as a no-op success sentinel) is not a failure and must not be treated as one."""
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
    if has_capability(conn, "text_encode"): return await call_capability(conn, "text_encode", payload, timeout_s=9000.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=9000.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/encode", json=payload)
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e: return {"error": str(e)}

async def flux2_generate(conn, payload: dict) -> dict:
    """Blocking on the image node (single GPU, serializes naturally) - caller's queue worker holds this until it returns."""
    if has_capability(conn, "image_generate"): return await call_capability(conn, "image_generate", payload, timeout_s=3600.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=3600.0, write=30.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/generate", json=payload)
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e: return {"error": str(e)}

async def flux2_generate_sequence(conn: dict, payload: dict) -> dict:
    if has_capability(conn, "image_sequence"): return await call_capability(conn, "image_sequence", payload, timeout_s=3600.0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=3600.0, write=30.0, pool=10.0)) as c:
        r = await c.post(f"{_base(conn)}/generate/sequence", json=payload)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text[:300]}"}

async def flux2_system_status(conn) -> dict:
    if has_capability(conn, "image_status"): return await call_capability(conn, "image_status", {}, timeout_s=8.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=4.0, read=8.0, write=4.0, pool=4.0)) as c:
            r = await c.get(f"{_base(conn)}/system/status")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_system_load(conn, model_path: str, vae_path: str) -> dict:
    """/system/load switches the node to model_path (it unloads what it holds first, and keeps it when model_path does not exist); while a generation runs it answers at once with status Error."""
    if has_capability(conn, "image_load"): return await call_capability(conn, "image_load", {"model_path": model_path, "vae_path": vae_path}, timeout_s=300.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/system/load", params={"model_path": model_path, "vae_path": vae_path})
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_system_unload(conn) -> dict:
    if has_capability(conn, "image_unload"): return await call_capability(conn, "image_unload", {}, timeout_s=30.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/system/unload")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def flux2_system_stop(conn) -> dict:
    if has_capability(conn, "image_stop"): return await call_capability(conn, "image_stop", {}, timeout_s=15.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)) as c:
            r = await c.post(f"{_base(conn)}/system/stop")
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e: return {"error": str(e)}

async def fetch_file(conn: dict, rel: str, dest) -> bool:
    """Copies one result file from the node to dest, for a node that shares no folder with this portal. False when the profile has no "file_fetch" or the node does not have the file. Written beside dest and renamed, so a cut connection leaves no half file."""
    spec, dest = _conn_profile(conn).get("file_fetch"), Path(dest)
    if not spec: return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)) as c:
        async with c.stream("GET", _base(conn) + spec["path"].replace("{path}", str(rel)), headers=_auth_headers(conn)) as r:
            if r.status_code != 200: return False
            with open(tmp, "wb") as f:
                async for chunk in r.aiter_bytes(): f.write(chunk)
    tmp.replace(dest)
    return True

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