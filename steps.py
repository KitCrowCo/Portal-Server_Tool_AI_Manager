"""steps.py - universal node type registry + implementations for ai_manager pipelines.
Every node type is one async function: fn(config: dict, data: dict, ctx: NodeContext) -> dict.
It reads whatever it needs from the shared pipeline `data` object - ctx.get()/ctx.resolve() are convenience helpers over the node's own key_map, not an access restriction;
the in_keys/out_keys a type declares are a scheduling contract, not a firewall - and returns a dict of LOGICAL output names -> values.
The engine remaps those to actual pipeline keys via the node's key_map and merges them into `data`.
"""
import re, json, sys, asyncio, uuid, time, io, wave, base64
from pathlib import Path
from tools.ai_manager import engine, resources, connections
from tools.ai_manager.connections import get_conn, lightrag_query, lightrag_insert_text, lightrag_list_entities, stream_llm, flux2_encode, flux2_generate, list_models_sync, list_conns, conns_matching

_NODE_TYPES: dict = {}
_EMBED_PATTERNS = ("embed", "minilm", "bge-", "gte-", "e5-", "nomic-embed", "arctic-embed")
ENV: dict = {}

def init(env: dict):
    global ENV
    ENV = env

def register_node_type(name, fn, label="", in_keys=None, out_keys=None, config_schema=None, guide=""): _NODE_TYPES[name] = {"fn": fn, "label": label or name, "in_keys": in_keys or [], "out_keys": out_keys or [], "config_schema": config_schema or [], "guide": guide}
def get_node_type(name: str) -> dict: return _NODE_TYPES.get(name)
def list_node_types() -> list: return [{"type": k, **{kk: vv for kk, vv in v.items() if kk != "fn"}} for k, v in _NODE_TYPES.items()]

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_FORCED_CUT_MARK = "[...chunk split - no sentence boundary found, cut forced here...]"

def _pack(units: list, target: int, overlap: int = 0) -> list:
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > target:
            chunks.append(cur)
            cur = (cur[-overlap:] + " " + u) if overlap else u
        else:
            cur = f"{cur} {u}" if cur else u
    if cur: chunks.append(cur)
    return chunks

def _hard_split(text: str, target: int) -> list:
    """Last-resort character split for a run-on with no usable punctuation at all - marks every forced cut so a downstream reader (human or LLM) can see a guess was forced, rather than trusting what looks like a clean break but isn't one."""
    out = []
    for i in range(0, len(text), target):
        piece = text[i:i+target]
        out.append(piece + (f" {_FORCED_CUT_MARK}" if i + target < len(text) else ""))
    return out

def _chunk_paragraphs(text: str, target_chars: int, overlap_chars: int = 0, hard_split_ratio: float = 2.0) -> list:
    """Tries paragraph boundaries first, falls back to sentence boundaries for an oversized paragraph, and only hard-splits (marked, not silent) when a single run-on still exceeds target_chars * hard_split_ratio with no punctuation to split on at all.
    This is deliberately NOT a rule that guarantees clean breaks - raw notes have too many edge cases for that to be worth hand-coding; the goal is graceful degradation with a visible flag, not silent failure."""
    segments = []
    for p in (p for p in text.split("\n\n") if p.strip()):
        if len(p) <= target_chars:
            segments.append(p); continue
        for seg in _pack(_SENTENCE_SPLIT_RE.split(p), target_chars):
            segments.extend(_hard_split(seg, target_chars) if len(seg) > target_chars * hard_split_ratio else [seg])
    return _pack(segments, target_chars, overlap_chars)

class NodeContext:
    """Wraps one node's own key_map so a type's implementation only ever deals in its own logical names, never in actual pipeline key strings - the same function works unmodified no matter what keys a pipeline instance wires it to."""
    def __init__(self, node, data, job_id, username, pool_cfg, depth=0):
        self.node, self.data, self.job_id, self.username, self.pool_cfg, self.depth = node, data, job_id, username, pool_cfg, depth

    def get(self, logical: str, default=""): return self.data.get(self.node.in_key(logical), default)

    def resolve(self, template) -> str:
        """{key} substitution against the shared data object using ACTUAL key names - for template config fields where the person types real pipeline key names directly."""
        if not isinstance(template, str): template = "" if template is None else str(template)
        def _sub(m):
            v = self.data.get(m.group(1))
            return str(v) if v is not None else ""
        return re.sub(r"\{(\w+)\}", _sub, template)

    def resolve_value(self, template):
        """Like resolve(), but a template that is EXACTLY one bare {key} reference returns the raw underlying value (list/dict/number, not stringified).
        Any other template shape (mixed text, multiple keys) still goes through normal string substitution - this only changes behavior for the single-reference case."""
        if isinstance(template, str):
            m = re.fullmatch(r"\{(\w+)\}", template.strip())
            if m: return self.data.get(m.group(1))
        return self.resolve(template)

    async def progress(self, message): await ENV["push_to_client"](self.username, {"t":"pipeline_event","job_id":self.job_id,"event":"running","payload":{"node":self.node.id,"message":message}})
    async def stream(self, key, delta): await ENV["push_to_client"](self.username, {"t":"pipeline_stream","job_id":self.job_id,"node":self.node.id,"key":key,"delta":delta})

# --- generate: any AI generation call - text or image, picked by modality ---

def _extract_json_fields(text: str, fields: list) -> dict | None:
    cleaned = re.sub(r'\A```(?:json)?\s*|\s*```\Z', '', text.strip())
    m = re.search(r'\{.*\}', cleaned, re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    found = {k: obj[k] for k in fields if k in obj}
    return found or None

async def node_generate(config: dict, data: dict, ctx: NodeContext) -> dict:
    modality = config.get("modality", "text")
    priority = config.get("priority") or ctx.pool_cfg.get("priority", "balanced")
    tags = [t.strip() for t in str(config.get("cnode_tags","")).split(",") if t.strip()]
    t0 = time.time()

    if modality == "image":
        _, enc_conn = _pick_typed_conn(config, ctx, "flux2_text", "generate(image) encoder")
        cnode, img_conn = _pick_typed_conn(config, ctx, "flux2_image", "generate(image)", pin_field="image_conn_id")
        prompt = ctx.resolve(config.get("user_template") or "{input}")
        job_tag = f"{ctx.job_id}_{uuid.uuid4().hex[:6]}"
        enc = await flux2_encode(enc_conn, prompt, job_id=job_tag, max_sequence_length=int(_opt(config, "max_sequence_length", 512)))
        if enc.get("error"): raise RuntimeError(f"""generate(image) encode: {enc['error']}""")
        init, mask, ref = (_blank_none(ctx.get(k, None)) for k in ("init_image", "mask_image", "reference_image"))   # optional in-keys: wire them through Extra In Keys / Key Map when a pipeline supplies them
        payload = {"prompt": prompt, "embed_job_id": job_tag, "width": int(_opt(config, "width", 0 if init else 1024)), "height": int(_opt(config, "height", 0 if init else 1024)), "steps": int(_opt(config, "steps", 4)), "guidance_scale": float(_opt(config, "cfg", 1.0)), "shift": float(_opt(config, "shift", 1.0)), "seed": int(_opt(config, "seed", -1))}   # with an init image a blank size is sent as 0 = the init image's own size
        if _opt(config, "image_model"): payload["model_path"] = config["image_model"]
        if init: payload.update({"image" if str(init).startswith("data:") else "image_path": init, "mask_image": mask or "", "strength": float(_opt(config, "strength", 0.75))})
        if ref: payload["reference_image"] = ref
        gen = await flux2_generate(img_conn, payload)
        if gen.get("error"): raise RuntimeError(f"""generate(image): {gen['error']}""")
        if cnode: resources.log_usage(cnode["id"], "generate:image", time.time()-t0)
        return {"file_name": gen["file_name"], "seed": gen.get("seed")}

    if modality == "video":
        start, end = _blank_none(ctx.get("start_image", None)), _blank_none(ctx.get("end_image", None))
        wanted = "start_end_to_video" if (start and end) else "image_to_video" if start else "text_to_video"
        if config.get("conn_id"): cnode, conn = _pick_capability_conn(config, ctx, wanted, "generate(video)")   # a pinned connection must declare exactly what the given frames ask for
        else:
            cnode = conn = None
            for cap in _VIDEO_CAPS[_VIDEO_CAPS.index(wanted):]:   # the pool may fall back to a lesser capability only: start + end -> start -> text
                try: cnode, conn = _pick_capability_conn(config, ctx, cap, "generate(video)"); break
                except RuntimeError: continue
            if not conn: raise RuntimeError(f"""generate(video): no connection in this node's resource pool declares '{wanted}' or a lesser video capability (check pool tags, or pin a connection)""")
            if cap != wanted: await ctx.progress(f"""no connection in the pool declares {wanted} - using {cap}, so the {"end frame" if wanted == "start_end_to_video" else "start frame"} is not used""")
            wanted = cap
        payload = {"prompt": ctx.resolve(config.get("user_template") or "{input}"), **{k: v for k, v in {"model": _opt(config, "video_model"), "negative_prompt": _opt(config, "negative_prompt"), "width": _opt(config, "video_width"), "height": _opt(config, "video_height"), "num_frames": _opt(config, "num_frames"), "fps": _opt(config, "fps"), "steps": _opt(config, "video_steps"), "guidance_scale": _opt(config, "video_cfg"), "scheduler": _opt(config, "scheduler"), "seed": _opt(config, "seed"), "offload_mode": _opt(config, "offload_mode"), "name": ctx.resolve(config["name_template"]) if _opt(config, "name_template") else None}.items() if v is not None}}   # unset values are left out, so the node applies the model's own defaults
        formats = [f.strip() for f in str(_opt(config, "formats", "")).split(",") if f.strip()]
        if formats: payload["formats"] = formats
        if wanted != "text_to_video": payload["start_image"] = start
        if wanted == "start_end_to_video": payload["end_image"] = end
        gen = await connections.call_capability(conn, wanted, payload, timeout_s=float(conn.get("values", {}).get("timeout_s") or 14400))   # clips take minutes to hours; a connection saved before its profile had timeout_s still gets 4 h
        if gen.get("error"): raise RuntimeError(f"""generate(video): {gen['error']}""")
        if cnode: resources.log_usage(cnode["id"], f"generate:{wanted}", time.time()-t0)
        return {"file_name": gen["file_name"], "frames_dir": gen.get("frames_dir", ""), "files": gen.get("files", {}), "frame_count": gen.get("frame_count"), "seed": gen.get("seed"), "capability_used": wanted}

    conn, cnode = (get_conn(config.get("conn_id","")), None) if config.get("conn_id") else (None, None)
    if not conn:
        picked = resources.pick_conn_for_capability(resources.resolve_candidates(ctx.pool_cfg, tags, capability="chat"), "chat", priority)
        if picked: cnode, conn = picked
        elif not (ctx.pool_cfg.get("whitelist_tags") or ctx.pool_cfg.get("whitelist_cnodes") or tags) and conns_matching("chat"): conn = conns_matching("chat")[0]
        else: raise RuntimeError("generate: no connection matches this node's resource pool (check pipeline/node whitelist-blacklist tags)")
    model = config.get("model","")
    if not model:
        models = list_models_sync(conn)
        model = pick_default_chat_model(models)
        if not model: raise RuntimeError(f"generate: connection {conn.get('_id','?')} has no non-embedding models available - pin one explicitly via the Model field. Available: {models}")
    seed_kwargs = {} if config.get("seed") in (None,"",-1) else {"seed": int(config["seed"])}
    raw_opts = config.get("enforce_options","")
    options = [o.strip() for o in raw_opts.split(",") if o.strip()] if isinstance(raw_opts,str) else (raw_opts or [])
    json_fields = [f.strip() for f in str(config.get("json_fields","")).split(",") if f.strip()]
    sys_p = ctx.resolve(config.get("system_prompt") or "")
    if options: sys_p = (sys_p + f"\n\nRespond with exactly one of these words and nothing else: {', '.join(options)}").strip()
    if json_fields: sys_p = (sys_p + f"\n\nRespond ONLY with a single JSON object with exactly these keys: {json.dumps(json_fields)}. No markdown fences, no text before or after the JSON.").strip()
    messages = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content": ctx.resolve(config.get("user_template") or "{input}")}]
    full, thinking_full = "", ""
    speaker = _SentenceSpeaker(ctx, config) if config.get("speak_while_writing") else None
    try:
        async for text, thinking in stream_llm(conn, messages, model, think=config.get("think", False), temperature=config.get("temperature", 0.7), num_ctx = int(resources.resolve_bound(config.get("num_ctx_mode","exact"), config.get("num_ctx", 16384), config.get("num_ctx_max"), fallback=16384)), num_predict=config.get("num_predict", -1), kv_cache_type=config.get("kv_cache_type") or None, cache_session=config.get("cache_session") or None, **seed_kwargs):
            full += text; thinking_full += thinking
            await ctx.stream("text", text)
            if speaker and text: speaker.feed(text)
    except BaseException:
        if speaker: speaker.cancel()
        ctx.partial = {"text": full, "thinking": thinking_full, **({"spoken_segments": speaker.seq, "audio_b64": speaker.joined()} if speaker else {})}   # kept by the engine when the node is stopped mid-reply
        raise
    if speaker: await speaker.close(cut=int(config.get("num_predict") or -1) > 0 and len(full) >= int(config["num_predict"]) * 3.6)   # about 3.6 characters per token: a reply this long most likely stopped at the token cap
    if cnode: resources.log_usage(cnode["id"], "generate:text", time.time()-t0)
    result = {"text": full, "thinking": thinking_full, **({"spoken_segments": speaker.seq, "audio_b64": speaker.joined()} if speaker else {})}
    if json_fields:
        parsed = _extract_json_fields(full, json_fields)
        if parsed: result.update(parsed)
        else: await ctx.progress("json_fields requested but parsing failed - raw text kept under 'text'")
    if options: result["choice"] = next((o for o in options if o.lower() in full.lower()), options[0])
    return result

# --- transform: deterministic data shaping, no AI call, no resource pool ---

_SAFE_BUILTINS = {"len": len, "str": str, "int": int, "float": float, "bool": bool, "min": min, "max": max, "abs": abs, "sum": sum, "any": any, "all": all, "sorted": sorted, "round": round, "list": list, "dict": dict, "tuple": tuple, "zip": zip, "range": range, "enumerate": enumerate, "reversed": reversed, "isinstance": isinstance}

async def node_transform(config: dict, data: dict, ctx: NodeContext) -> dict:
    mode = config.get("mode", "template")
    if mode == "template": return {"value": ctx.resolve(config.get("template", "{input}"))}
    if mode == "expr":
        variables = {name: ctx.resolve_value(tpl) for name, tpl in _parse_json_config(config.get("vars_json"), {}).items()}
        try: value = eval(config.get("expr","input"), {"__builtins__": _SAFE_BUILTINS, **variables, "input": ctx.get("input")})   # variables as globals: comprehensions cannot see eval locals before Python 3.12
        except Exception as e: raise RuntimeError(f"transform(expr): {e}")
        return {"value": value}
    if mode == "regex_find":
        pattern = config.get("pattern","")
        if not pattern: raise RuntimeError("transform(regex_find): pattern required")
        content = ctx.get("input")
        flags = re.DOTALL if config.get("dotall") else 0
        matches = [{"text": m.group(0), "groups": list(m.groups()), "start": m.start(), "end": m.end()} for m in re.finditer(pattern, content, flags)]
        return {"matches": matches, "count": len(matches)}
    if mode == "regex_replace":
        source = ctx.get("input")
        def _sub(m):
            r = config.get("replacement_template","").replace("{match}", m.group(0))
            for i, g in enumerate(m.groups() or []): r = r.replace(f"{{{i}}}", g or "")
            return r
        return {"value": re.sub(config.get("pattern",""), _sub, source, count=(0 if config.get("replace_mode","all")=="all" else 1))}
    if mode == "chunk":
        text = ctx.get("input")
        target = int(config.get("chunk_chars", 4000) or 4000)
        overlap = int(config.get("overlap_chars", 0) or 0)
        chunks = _chunk_paragraphs(text, target, overlap)
        return {"chunks": chunks, "count": len(chunks)}
    if mode == "python":
        if config.get("script_body"):
            tmp = Path(f"./data/ai_manager/_scratch_scripts/{ctx.job_id}_{ctx.node.id}.py")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(ctx.resolve(config["script_body"]))
            script_path = str(tmp)
        else:
            script_path = config.get("script_path","")
        if not script_path or not Path(script_path).is_file(): raise RuntimeError(f"transform(python): script not found: {script_path}")
        arg = ctx.resolve(config.get("input_template", "{input}"))
        timeout_s = int(config.get("timeout_s", 600) or 600)
        proc = await asyncio.create_subprocess_exec(sys.executable, script_path, arg, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try: stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill(); await proc.communicate()
            raise RuntimeError(f"transform(python): timed out after {timeout_s}s")
        if proc.returncode != 0: raise RuntimeError(f"transform(python): exit {proc.returncode}\n{stderr.decode(errors='replace')[-1000:]}")
        out = stdout.decode(errors="replace").strip()
        try: parsed = json.loads(out.splitlines()[-1]) if out else {}
        except Exception: parsed = {"value": out}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    raise RuntimeError(f"transform: unknown mode '{mode}'")

# --- file io ---

async def node_file_read(config: dict, data: dict, ctx: NodeContext) -> dict:
    fm = ENV["tools"]["built_ins"].FileManager(config.get("fm_root") or "./data/_common")
    path = ctx.resolve(config.get("path") or "") or ctx.get("path")
    if not path: raise RuntimeError("file_read: resolved path is empty")
    return {"text": fm.read(path)}

async def node_file_write(config: dict, data: dict, ctx: NodeContext) -> dict:
    bi = ENV["tools"]["built_ins"]
    fm_root = config.get("fm_root") or "./data/_common"
    fm = bi.FileManager(fm_root)
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(fm_root) / "_shadow"), auto_accept=bool(config.get("auto_accept")))
    path = ctx.resolve(config.get("path") or "")
    if not path.strip(): raise RuntimeError("file_write: resolved path is empty")
    content = ctx.get("content")
    if content == "" and not config.get("allow_empty"): raise RuntimeError("file_write: content is empty - check this node's key_map/in-keys")
    if config.get("binary"):
        entry = shadow.stage_binary(path, content if isinstance(content, (bytes, bytearray)) else Path(content).read_bytes(), author=f"pipeline:{ctx.job_id}")
    else:
        entry = shadow.stage(path, str(content), author=f"pipeline:{ctx.job_id}")
    return {"path": path, "status": entry["status"]}

async def node_file_list(config: dict, data: dict, ctx: NodeContext) -> dict:
    exts = tuple(e.strip().lower() for e in (config.get("extensions","") or "").split(",") if e.strip())
    root = Path(config.get("root", "./data/ai_tools/_knowledge"))
    return {"files": sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.is_file() and (not exts or f.suffix.lower() in exts))}

# --- knowledge: query / insert / entities against LightRAG, one node ---

async def node_knowledge(config: dict, data: dict, ctx: NodeContext) -> dict:
    mode = config.get("mode", "query")
    priority = config.get("priority") or ctx.pool_cfg.get("priority", "balanced")
    tags = [t.strip() for t in str(config.get("cnode_tags","")).split(",") if t.strip()]
    conn, cnode = (get_conn(config.get("conn_id",""), conn_type="lightrag"), None) if config.get("conn_id") else (None, None)
    if not conn:
        picked = resources.pick_conn_for_capability(resources.resolve_candidates(ctx.pool_cfg, tags, capability="knowledge_query"), "knowledge_query", priority)
        if not picked: raise RuntimeError("knowledge: no lightrag connection matches this node's resource pool")
        cnode, conn = picked
    t0 = time.time()
    if mode == "insert":
        res = await lightrag_insert_text(conn, ctx.resolve(config.get("text_template", "{input}")), config.get("source_label",""))
        if cnode: resources.log_usage(cnode["id"], "knowledge:insert", time.time()-t0)
        return res if isinstance(res, dict) else {"status": res}
    if mode == "entities":
        entities = await lightrag_list_entities(conn, limit=config.get("limit", 500))
        if cnode: resources.log_usage(cnode["id"], "knowledge:entities", time.time()-t0)
        return {"entities": entities}
    r = await lightrag_query(conn, ctx.resolve(config.get("query_template", "{input}")), config.get("query_mode", "hybrid"))
    if cnode: resources.log_usage(cnode["id"], "knowledge:query", time.time()-t0)
    return {"response": r.get("response", r.get("error","")), "raw": r}

# --- pipeline composition ---

def _pipeline_options(values=None): return [("", "(none)")] + [(p["id"], p.get("name",p["id"])) for p in engine.list_pipelines()]

async def node_pipeline(config: dict, data: dict, ctx: NodeContext) -> dict:
    pid = config.get("pipeline_id","")
    if not pid: raise RuntimeError("pipeline: no pipeline selected")
    import_keys = [k.strip() for k in str(config.get("import_keys","")).split(",") if k.strip()]
    export_keys = [k.strip() for k in str(config.get("export_keys","")).split(",") if k.strip()]
    sub_inputs = {k: data.get(k, "") for k in import_keys}
    sub_job_id = f"job_{uuid.uuid4().hex[:10]}"
    sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1, job_id=sub_job_id)
    return {**{k: sub_data.get(k, "") for k in export_keys}, "_sub_job_id": sub_job_id}

def _sub_run_error(job_id: str) -> str:
    """The first failed node's message of a finished sub-run, or "" when it succeeded."""
    job = engine.load_job(job_id) or {}
    return next((n.get("message") or f"""node {n.get('name') or n.get('id')} failed""" for n in job.get("flow", {}).get("nodes", []) if n.get("status") == "error"), "job failed" if job.get("status") == "error" else "")

async def node_pipeline_foreach(config: dict, data: dict, ctx: NodeContext) -> dict:
    items = ctx.get("items")
    if isinstance(items, str):
        try: items = json.loads(items)
        except Exception: items = [x.strip() for x in items.split("\n") if x.strip()]
    if not isinstance(items, list) or not items: raise RuntimeError("pipeline_foreach: items resolved to an empty list")
    pid = config.get("pipeline_id","")
    if not pid: raise RuntimeError("pipeline_foreach: no pipeline selected")
    item_key = config.get("item_key","item")
    import_keys = [k.strip() for k in str(config.get("import_keys","")).split(",") if k.strip()]
    export_keys = [k.strip() for k in str(config.get("export_keys","")).split(",") if k.strip()]
    results, sub_job_ids = [], []
    for i, item in enumerate(items):
        await ctx.progress(f"item {i+1}/{len(items)}")
        sub_inputs = {item_key: item, **{k: data.get(k, "") for k in import_keys}}
        sub_job_id = f"job_{uuid.uuid4().hex[:10]}"
        sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1, job_id=sub_job_id)
        sub_job_ids.append(sub_job_id)
        err = _sub_run_error(sub_job_id)
        if err and config.get("on_item_error") == "stop": raise RuntimeError(f"pipeline_foreach: item {i+1}/{len(items)} failed - {err}")
        results.append({**{k: sub_data.get(k, "") for k in export_keys}, **({"_error": err} if err else {})})
    return {"results": results, "count": len(results), "_sub_job_ids": sub_job_ids}

async def node_pipeline_reduce(config: dict, data: dict, ctx: NodeContext) -> dict:
    """Like pipeline_foreach but SEQUENTIAL: threads an accumulator value through each item in order, so the sub-pipeline can build on its own prior output as it goes (a running summary, a running edited document).
    pipeline_foreach's per-item calls are independent and cannot see each other's results - use this instead whenever the processing order matters."""
    items = ctx.get("items")
    if isinstance(items, str):
        try: items = json.loads(items)
        except Exception: items = [x.strip() for x in items.split("\n") if x.strip()]
    if not isinstance(items, list) or not items: raise RuntimeError("pipeline_reduce: items resolved to an empty list")
    pid = config.get("pipeline_id","")
    if not pid: raise RuntimeError("pipeline_reduce: no pipeline selected")
    item_key = config.get("item_key","item")
    acc_key = config.get("accumulator_key","accumulator")
    acc_export_key = config.get("accumulator_export_key") or acc_key
    import_keys = [k.strip() for k in str(config.get("import_keys","")).split(",") if k.strip()]
    accumulator = ctx.get("accumulator")
    sub_job_ids = []
    for i, item in enumerate(items):
        await ctx.progress(f"item {i+1}/{len(items)}")
        sub_inputs = {item_key: item, acc_key: accumulator, **{k: data.get(k, "") for k in import_keys}}
        sub_job_id = f"job_{uuid.uuid4().hex[:10]}"
        sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1, job_id=sub_job_id)
        accumulator = sub_data.get(acc_export_key, accumulator)
        sub_job_ids.append(sub_job_id)
    return {"accumulator": accumulator, "count": len(items), "_sub_job_ids": sub_job_ids}

async def node_branch(config: dict, data: dict, ctx: NodeContext) -> dict:
    """Decides a value (expression or LLM), looks it up in Routes, and calls whichever sub-pipeline matches."""
    if config.get("decide_mode", "expr") == "llm":
        decision = (await node_generate({**config, "enforce_options": config.get("options",""), "user_template": config.get("decide_template","{input}")}, data, ctx)).get("choice","")
    else:
        var_templates = _parse_json_config(config.get("vars_json"), {})
        local_vars = {name: ctx.resolve_value(tpl) for name, tpl in var_templates.items()}
        try: decision = str(eval(config.get("decide_expr","input"), {"__builtins__": {"len":len,"str":str,"int":int}}, {**local_vars, "input": ctx.get("input")}))
        except Exception as e: raise RuntimeError(f"branch: decide_expr failed: {e}")
    routes = _parse_json_config(config.get("routes_json"), {})
    pid = routes.get(decision) or config.get("default_pipeline_id","")
    if not pid: raise RuntimeError(f"branch: no route for decision '{decision}' and no default_pipeline_id set")
    import_keys = [k.strip() for k in str(config.get("import_keys","")).split(",") if k.strip()]
    export_keys = [k.strip() for k in str(config.get("export_keys","")).split(",") if k.strip()]
    sub_inputs = {k: data.get(k, "") for k in import_keys}
    sub_job_id = f"job_{uuid.uuid4().hex[:10]}"
    sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1, job_id=sub_job_id)
    return {**{k: sub_data.get(k, "") for k in export_keys}, "decision": decision, "_sub_job_id": sub_job_id}

# --- capability connection resolution (shared by the speech and agent node types) ---

def _node_tags(config: dict, field: str = "cnode_tags") -> list: return [t.strip() for t in str(config.get(field) or "").split(",") if t.strip()]

def _pick_capability_conn(config: dict, ctx: NodeContext, capability: str, label: str, pin_field: str = "conn_id", tags_field: str = "cnode_tags") -> tuple:
    """Returns (cnode or None, conn) for one capability. A pinned connection wins and must declare the capability; otherwise the node's pool (pipeline pool narrowed by this node's tags) picks by Priority.
    With no pool restrictions at all and no CNode carrying the capability, falls back to the first connection declaring it - the same fallback generate uses for chat."""
    if config.get(pin_field):
        conn = get_conn(config[pin_field])
        if not conn: raise RuntimeError(f"""{label}: pinned connection '{config[pin_field]}' no longer exists""")
        if not connections.has_capability(conn, capability): raise RuntimeError(f"""{label}: pinned connection '{config[pin_field]}' ({conn.get('connection_type')}) does not declare '{capability}'""")
        return None, conn
    tags = _node_tags(config, tags_field)
    picked = resources.pick_conn_for_capability(resources.resolve_candidates(ctx.pool_cfg, tags, capability=capability), capability, config.get("priority") or ctx.pool_cfg.get("priority", "balanced"))
    if picked: return picked
    if not (ctx.pool_cfg.get("whitelist_tags") or ctx.pool_cfg.get("whitelist_cnodes") or tags) and conns_matching(capability): return None, conns_matching(capability)[0]
    raise RuntimeError(f"""{label}: no connection in this node's resource pool declares '{capability}' (check pool tags, or pin a connection)""")

def _pick_typed_conn(config: dict, ctx: NodeContext, conn_type: str, label: str, pin_field: str = "conn_id") -> tuple:
    """(cnode or None, conn) for one connection TYPE (flux2_text, flux2_image): a pinned connection wins; otherwise the node's pool picks by Priority;
    with no pool restrictions at all and no CNode carrying that type, the first saved connection of the type - the fallback chat and speech use."""
    if config.get(pin_field):
        conn = get_conn(config[pin_field])
        if not conn: raise RuntimeError(f"""{label}: pinned connection '{config[pin_field]}' no longer exists""")
        return None, conn
    tags = _node_tags(config)
    picked = resources.pick_conn(resources.resolve_candidates(ctx.pool_cfg, tags, conn_type), conn_type, config.get("priority") or ctx.pool_cfg.get("priority", "balanced"))
    if picked: return picked
    if not (ctx.pool_cfg.get("whitelist_tags") or ctx.pool_cfg.get("whitelist_cnodes") or tags) and list_conns(conn_type): return None, list_conns(conn_type)[0]
    raise RuntimeError(f"""{label}: no {conn_type} connection in this node's resource pool (check pool tags, or pin a connection)""")

def _blank_none(v): return None if v in (None, "") else v   # an unset data key or blank form value means "not given"

def _opt(config: dict, key: str, default=None):
    """A config value, or default when the field was left blank - a blank number must fall back, never reach a node as "" or null."""
    v = config.get(key)
    return default if v in (None, "") else v

def _pick_chat_model(conn: dict, model: str, label: str) -> str:
    if model: return model
    models = list_models_sync(conn)
    picked = pick_default_chat_model(models)
    if not picked: raise RuntimeError(f"""{label}: connection {conn.get('_id','?')} has no non-embedding models available - pin one via the Model field. Available: {models}""")
    return picked

# --- speech: transcribe / speak through whichever connection declares speech_to_text / text_to_speech ---

_SENT_END = re.compile(r"\n+|(?<=[.!?\u2026])[\"')\]]*[ \t]+")   # a line break, or sentence punctuation (plus closing quotes/brackets) followed by a space

class _SentenceSpeaker:
    """Speaks a reply while it is still being written. Text is buffered until a sentence ends (a line break, or . ! ? and a space); each finished
    sentence goes to text-to-speech at once, one request at a time and in order, and its audio is pushed to the browser as a `cm-voice-audio`
    trigger {sid, turn, seq, audio_b64, format, text, last}. The browser queues the segments by seq and plays them back to back, so the first words
    are heard after one sentence instead of after the whole reply. A failed segment is pushed with empty audio so playback order never stalls.
    The spoken wav segments are also kept and joined into one wav (joined()), returned as the generate node's audio_b64 so the reply can be replayed."""
    def __init__(self, ctx: NodeContext, config: dict):
        self.ctx, self.sid, self.buf, self.seq, self.q, self.audio = ctx, ctx.resolve(str(config.get("speak_sid") or "")), "", 0, asyncio.Queue(), []
        self.min_chars, self.drop_tail = int(_opt(config, "speak_min_chars", 12)), bool(config.get("speak_drop_unfinished", True))
        self.payload = {"voice": config.get("speak_voice") or None, "length_scale": None if config.get("speak_length_scale") in (None, "") else float(config["speak_length_scale"])}
        _, self.conn = _pick_capability_conn({**config, "conn_id": config.get("speak_conn_id") or ""}, ctx, "text_to_speech", "generate(speak while writing)")
        self.worker = asyncio.create_task(self._work())

    def feed(self, text: str):
        self.buf += text
        while (m := _SENT_END.search(self.buf, self.min_chars if len(self.buf) > self.min_chars else len(self.buf))):   # very short pieces ("Yes." "Hi!") wait for the next sentence instead of costing a request each
            self.q.put_nowait(self.buf[:m.end()].strip()); self.buf = self.buf[m.end():]

    async def close(self, cut: bool = False):
        """Speaks what is left - unless the reply was cut at the token cap mid-sentence and drop_tail is set - then waits for every segment to be sent."""
        tail = self.buf.strip()
        if tail and not (cut and self.drop_tail and tail[-1] not in ".!?\u2026\"')]"): self.q.put_nowait(tail)
        self.q.put_nowait(None)
        await self.worker
        await self._push({"seq": self.seq, "last": True})

    def cancel(self): self.worker.cancel()

    def joined(self) -> str:
        """Every spoken segment so far as one wav (base64), for replay. Segments whose wav parameters differ from the first are left out."""
        if not self.audio: return ""
        out, first = io.BytesIO(), None
        with wave.open(out, "wb") as w:
            for b in self.audio:
                with wave.open(io.BytesIO(base64.b64decode(b.split(",", 1)[-1]))) as r:
                    if first is None: first = r.getparams(); w.setparams(first)
                    if r.getparams()[:3] == first[:3]: w.writeframes(r.readframes(r.getnframes()))
        return base64.b64encode(out.getvalue()).decode()

    async def _push(self, detail: dict): await ENV["push_to_client"](self.ctx.username, {"t": "trigger", "event": "cm-voice-audio", "detail": {"sid": self.sid, "turn": self.ctx.job_id, **detail}})

    async def _work(self):
        while (text := await self.q.get()) is not None:
            if not _words(text): continue   # punctuation-only pieces have nothing to say
            r = await connections.call_capability(self.conn, "text_to_speech", {"text": text, **self.payload})
            if r.get("error"): await self.ctx.progress(f"""speak while writing: segment {self.seq} failed - {r['error']}""")
            elif r.get("format", "wav") == "wav" and r.get("audio_b64"): self.audio.append(r["audio_b64"])
            await self._push({"seq": self.seq, "audio_b64": "" if r.get("error") else r.get("audio_b64", ""), "format": r.get("format", "wav"), "text": text, "last": False})
            self.seq += 1

def _words(text: str) -> str: return re.sub(r"[^\w']+", " ", str(text).lower()).strip()   # lowercase words only: punctuation and spacing never decide whether something was said

async def node_transcribe(config: dict, data: dict, ctx: NodeContext) -> dict:
    """Nothing said (an empty or punctuation-only transcript, an ignored phrase, or only segments the recognizer rates as non-speech) returns no 'text' key, so every node waiting on it ends unreached; the raw transcript goes to 'dropped_text'."""
    audio = ctx.get("audio_b64")
    if not audio: raise RuntimeError("transcribe: audio_b64 in-key is empty")
    cnode, conn = _pick_capability_conn(config, ctx, "speech_to_text", "transcribe")
    t0 = time.time()
    r = await connections.call_capability(conn, "speech_to_text", {"audio_b64": audio, "language": config.get("language") or None, "model": config.get("stt_model") or None, "vad_filter": bool(config.get("vad_filter", True)), "initial_prompt": ctx.resolve(config.get("initial_prompt") or "") or None})
    if r.get("error"): raise RuntimeError(f"""transcribe: {r['error']}""")
    if cnode: resources.log_usage(cnode["id"], "transcribe", time.time()-t0, {"audio_s": r.get("duration_s")})
    segs, cut = r.get("segments", []), _opt(config, "max_no_speech_prob")
    if cut is not None: segs = [s for s in segs if float(s.get("no_speech_prob") or 0) <= float(cut)]
    text = " ".join(str(s.get("text", "")).strip() for s in segs).strip() if cut is not None else r.get("text", "")
    heard = _words(text)
    out = {"segments": segs, "language": r.get("language", "")}
    if (config.get("skip_empty", True) and not heard) or heard in {_words(p) for p in str(config.get("ignore_phrases") or "").splitlines() if _words(p)}: return {**out, "dropped_text": r.get("text", "")}
    return {**out, "text": text}

async def node_speak(config: dict, data: dict, ctx: NodeContext) -> dict:
    text = str(ctx.get("text"))
    if not text.strip(): raise RuntimeError("speak: text in-key is empty")
    cnode, conn = _pick_capability_conn(config, ctx, "text_to_speech", "speak")
    t0 = time.time()
    r = await connections.call_capability(conn, "text_to_speech", {"text": text, "voice": config.get("voice") or None, "length_scale": None if config.get("length_scale") in (None, "") else float(config["length_scale"])})
    if r.get("error"): raise RuntimeError(f"""speak: {r['error']}""")
    if cnode: resources.log_usage(cnode["id"], "speak", time.time()-t0, {"audio_s": r.get("duration_s")})
    return {"audio_b64": r["audio_b64"], "format": r.get("format", "wav"), "duration_s": r.get("duration_s")}

async def node_identify_speaker(config: dict, data: dict, ctx: NodeContext) -> dict:
    """Who said it. The audio is embedded by a connection declaring speaker_embed and compared (cosine) with each enrolled profile in the 'profiles'
    in-key: [{name, code, role, embedding (unit length)}]. At or above match_threshold the best profile is the speaker; between guess_threshold and
    match_threshold it is a likely match (guess mark); below, or with no profiles, the speaker is other_name. speaker_tag is tag_format filled with
    {name} {code} {role} {guess}, ready to prefix the transcript. A failed embedding leaves everything blank and the turn goes on unlabeled."""
    profiles, blank = ctx.get("profiles") or [], {"speaker": "", "speaker_code": "", "speaker_role": "", "speaker_score": 0.0, "speaker_tag": ""}
    cnode, conn = _pick_capability_conn(config, ctx, "speaker_embed", "identify_speaker")
    t0 = time.time()
    r = await connections.call_capability(conn, "speaker_embed", {"audio_b64": ctx.get("audio_b64")})
    if r.get("error"): await ctx.progress(f"""identify_speaker: {r['error']} - turn left unlabeled"""); return blank
    if cnode: resources.log_usage(cnode["id"], "identify_speaker", time.time()-t0, {"audio_s": r.get("duration_s")})
    v = r["embedding"]
    best, score = max(((p, sum(a * b for a, b in zip(v, p["embedding"]))) for p in profiles if p.get("embedding")), key=lambda t: t[1], default=(None, 0.0))
    p = best if best and score >= float(_opt(config, "guess_threshold", 0.35)) else {"name": config.get("other_name") or "Other", "code": config.get("other_code") or "0000", "role": "unknown"}
    guess = config.get("guess_mark", "?") if best is p and score < float(_opt(config, "match_threshold", 0.5)) else ""
    return {"speaker": p["name"], "speaker_code": p.get("code", ""), "speaker_role": p.get("role", ""), "speaker_score": round(score, 3), "speaker_tag": (config.get("tag_format") or "({name}{guess}): ").format(name=p["name"], code=p.get("code", ""), role=p.get("role", ""), guess=guess)}

# --- call_capability: any capability a connection's profile declares (video_join, model_list, voice_list, ...) ---

def _resolve_deep(v, ctx: NodeContext):
    if isinstance(v, dict): return {k: _resolve_deep(x, ctx) for k, x in v.items()}
    if isinstance(v, list): return [_resolve_deep(x, ctx) for x in v]
    return ctx.resolve_value(v) if isinstance(v, str) else v

async def node_call_capability(config: dict, data: dict, ctx: NodeContext) -> dict:
    cap = str(config.get("capability") or "").strip()
    if not cap: raise RuntimeError("call_capability: no capability set")
    cnode, conn = _pick_capability_conn(config, ctx, cap, f"call_capability({cap})")
    payload = _resolve_deep(_parse_json_config(config.get("payload_json"), {}), ctx)
    t0 = time.time()
    r = await connections.call_capability(conn, cap, payload, timeout_s=float(_opt(config, "timeout_s", 0) or conn.get("values", {}).get("timeout_s") or 3600))
    if r.get("error"): raise RuntimeError(f"""call_capability({cap}): {r['error']}""")
    if cnode: resources.log_usage(cnode["id"], f"call_capability:{cap}", time.time()-t0)
    return {"result": r, **{f: r[f] for f in (x.strip() for x in str(config.get("out_fields") or "").split(",")) if f and f in r}}

def looks_like_embedding(model_name: str) -> bool: return any(p in model_name.lower() for p in _EMBED_PATTERNS)

def pick_default_chat_model(models: list) -> str:
    """Best-effort exclusion of obvious embedding-only models from an auto-pick default. A real capability check (per-model /api/show) would be more accurate but costs a network round-trip per candidate
    - this stays a fast heuristic; pin a model explicitly via the Model field whenever it matters."""
    candidates = [m for m in models if not looks_like_embedding(m)]
    return candidates[0] if candidates else ""

def _llm_conn_options(values=None): return [("", "(pool-resolved by priority)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in conns_matching("chat")]

_VIDEO_CAPS = ["start_end_to_video", "image_to_video", "text_to_video"]   # richest first; a pool pick may fall back rightward only

def _conn_label(c: dict) -> tuple: return (c["_id"], f"""{c.get("display_name", c["_id"])} [{c.get("connection_type", "?")}]""")

def _generate_conn_options(values=None):
    """Generate's Connection select follows Modality: chat connections for text, text encoders for image, video-capable connections for video."""
    m = (values or {}).get("modality") or "text"
    conns = list_conns("flux2_text") if m == "image" else [c for c in list_conns(get_all=True) if any(connections.has_capability(c, k) for k in _VIDEO_CAPS)] if m == "video" else conns_matching("chat")
    return [("", "(pool-resolved by priority)")] + [_conn_label(c) for c in conns]

def _tts_conn_options(values=None): return [("", "(pool-resolved by capability)")] + [_conn_label(c) for c in conns_matching("text_to_speech")]
def _image_node_options(values=None): return [("", "(pool-resolved by priority)")] + [_conn_label(c) for c in list_conns("flux2_image")]
def _all_conn_options(values=None): return [("", "(pool-resolved by capability)")] + [_conn_label(c) for c in list_conns(get_all=True)]

def _node_model_options(kind: str):
    """Options for image_model / video_model from the node's own model_list (blank = the node's default / loaded model). The saved value stays selectable when the node is unreachable."""
    def _opts(values=None):
        values = values or {}
        pin = values.get("image_conn_id" if kind == "image" else "conn_id")
        conn = get_conn(pin) if pin else next(iter(list_conns("flux2_image")), None)
        r = connections.call_capability_sync(conn, "model_list", {}, timeout_s=3.0) if conn else {}
        names = [m["name"] for m in r.get(kind, []) if m.get("modes")] if isinstance(r.get(kind), list) else []
        cur = values.get(f"{kind}_model") or ""
        return [("", "(node default)")] + [(n, n) for n in names] + ([(cur, f"{cur} (saved, not listed by the node)")] if cur and cur not in names else [])
    return _opts

def _model_options_for_pinned_conn(values=None):
    conn_id = (values or {}).get("conn_id","")
    if not conn_id: return [("", "(auto - non-embedding models only)")]
    conn = get_conn(conn_id)
    models = list_models_sync(conn) if conn else []
    cur = (values or {}).get("model","")
    opts = [("", "(auto)")] + [(m, m + (" - looks like embedding" if looks_like_embedding(m) else "")) for m in models]
    if cur and cur not in models: opts.append((cur, cur + " (saved, not currently listed)"))
    return opts

def _parse_json_config(val, default=None):
    """Config fields of type 'json' arrive already-parsed as a dict/list (the builder parses them once at save time) OR as a raw JSON string (hand-edited pipeline JSON, imports).
    Accept either without double-parsing - json.loads() on an already-parsed dict raises TypeError, which was previously swallowed and silently produced an empty fallback instead of the real variables."""
    if val is None or val == "": return default if default is not None else {}
    if isinstance(val, (dict, list)): return val
    try: return json.loads(val)
    except Exception: return default if default is not None else {}

# --- registration ---

def register_builtins():
    BI = ENV["tools"]["built_ins"]
    def _key_map_field(): return BI.SettingField("key_map", "Key Map (JSON: logical -> actual pipeline key)", type="json", default={}, advanced=True, hint='Only needed to rename this node\'s in/out keys, e.g. {"input":"user_query","text":"draft"}. Unmapped logical names pass through unchanged.')

    def _conn_options_for_type(conn_type):
        def _opts(values=None): return [("", "(pool-resolved by priority)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in conns_matching(conn_type)]
        return _opts

    def _pool_fields(include_model=True, conn_type="chat"):
        fields = [BI.SettingField("cnode_tags", "Resource Pool Tags (comma-sep)", type="text", advanced=True, hint="Narrows the pipeline's own pool to CNodes carrying ALL these tags. Blank = use the whole pipeline pool."),
                  BI.SettingField("priority", "Priority", type="select", default="", options=[("","(inherit pipeline default)"),("speed","Speed"),("balanced","Balanced"),("quality","Quality")], advanced=True),
                  BI.SettingField("conn_id", "Connection", type="select", default="", options=_conn_options_for_type(conn_type), hint="Full pool auto-selection (multi-candidate scoring) is planned but not yet built - pin a connection here until then.")]
        if include_model: fields.append(BI.SettingField("model", "Model", type="select", default="", options=_model_options_for_pinned_conn, hint="Populates once a connection is pinned above. Leave blank to auto-pick the first non-embedding model."))
        return fields

    register_node_type("generate", node_generate, "Generate (text, image or video)", in_keys=["input"], out_keys=["text","thinking"], config_schema=[
        BI.SettingField("modality","Modality","select",default="text", options=[("text","Text"),("image","Image"),("video","Video")], hint="Image reads optional init_image / mask_image / reference_image keys; video reads optional start_image / end_image keys (add them to Extra In Keys when a pipeline supplies them)."),
        BI.SettingField("system_prompt","System Prompt","textarea",default="You are a helpful AI assistant."),
        BI.SettingField("user_template","User/Prompt Template","textarea",default="{input}", hint="{key} substitutes real pipeline keys directly."),
        BI.SettingField("temperature","Temperature","number",default=0.7),
        BI.SettingField("num_ctx_mode","Context Window Mode","select",default="exact",options=[("exact","Exact"),("at_least","At least"),("no_more_than","No more than"),("range","Range (min-max)")],advanced=True),
        BI.SettingField("num_ctx","Context Window Tokens","number",default=16384,step=1),
        BI.SettingField("num_ctx_max","Context Window Max (range mode only)","number",default=None,step=1,advanced=True),
        BI.SettingField("num_predict","Max Output Tokens","number",default=-1,step=1),
        BI.SettingField("think","Thinking Effort","select",default="medium", options=[("","Off"),("low","Low"),("medium","Medium"),("high","High")]),
        BI.SettingField("enforce_options","Enforced Options (comma-sep)","text",advanced=True),
        BI.SettingField("seed","Seed (blank/-1 = random)","number",default=None,advanced=True,step=1),
        BI.SettingField("width","Width (image)","number",default=1024,step=1,advanced=True),
        BI.SettingField("height","Height (image)","number",default=1024,step=1,advanced=True),
        BI.SettingField("steps","Steps (image)","number",default=4,step=1,advanced=True),
        BI.SettingField("cfg","Guidance Scale (image)","number",default=1.0,advanced=True),
        BI.SettingField("strength","Init Image Strength (image)","number",default=0.75,advanced=True,hint="With an init image: 1 repaints fully, lower keeps more of it. Width/height blank = the init image's own size."),
        BI.SettingField("image_model","Image Model (image)","select",default="",options=_node_model_options("image"),advanced=True),
        BI.SettingField("image_conn_id","Image Node Connection (image)","select",default="",options=_image_node_options,advanced=True,hint="For image modality the Connection field above is the text encoder; this is the node that renders."),
        BI.SettingField("video_model","Video Model (video)","select",default="",options=_node_model_options("video"),advanced=True,hint="Blank = the loaded video model when it can do the job, else the first that can."),
        BI.SettingField("negative_prompt","Negative Prompt (video)","textarea",advanced=True,hint="Blank = the node's default."),
        BI.SettingField("video_width","Width (video)","number",default=None,step=1,advanced=True,hint="Blank = the model's default; a Fun InP model with a start frame keeps that frame's shape."),
        BI.SettingField("video_height","Height (video)","number",default=None,step=1,advanced=True),
        BI.SettingField("num_frames","Frames (video)","number",default=None,step=1,advanced=True,hint="Rounded up to what the model can make (4k+1)."),
        BI.SettingField("fps","Frames per Second (video)","number",default=None,advanced=True),
        BI.SettingField("video_steps","Steps (video)","number",default=None,step=1,advanced=True),
        BI.SettingField("video_cfg","Guidance Scale (video)","number",default=None,advanced=True),
        BI.SettingField("scheduler","Sampler (video)","select",default="",options=[("","(model default)"),("ddim","DDIM"),("cog_ddim","CogVideoX DDIM"),("cog_dpm","CogVideoX DPM")],advanced=True),
        BI.SettingField("formats","Formats (video, comma-sep)","text",default="",advanced=True,hint="gif, webp, mp4. Blank = gif. Joining clips works from the frames every clip keeps."),
        BI.SettingField("offload_mode","Memory Offload (video)","select",default="",options=[("","(none)"),("sequential","Sequential blocks (slowest, least memory)")],advanced=True),
        BI.SettingField("name_template","Output Name (video)","text",default="",advanced=True,hint="File name prefix; supports {key}."),
        BI.SettingField("speak_while_writing","Speak While Writing (text)","checkbox",default=False,advanced=True,hint="Each finished sentence is spoken while the rest is still being written; the audio goes to the chat named below as cm-voice-audio segments. Tell the model to end sentences with a line break for the earliest start."),
        BI.SettingField("speak_conn_id","Speech Connection (speak while writing)","select",default="",options=_tts_conn_options,advanced=True),
        BI.SettingField("speak_voice","Voice (speak while writing)","text",advanced=True,hint="Blank = the speech node's default."),
        BI.SettingField("speak_length_scale","Speaking Pace (speak while writing)","number",default=None,advanced=True),
        BI.SettingField("speak_sid","Chat Id to Play In (speak while writing)","text",advanced=True,hint="The browser chat that plays the segments; supports {key}."),
        BI.SettingField("speak_min_chars","Shortest Spoken Piece (chars)","number",default=12,step=1,advanced=True,hint="Shorter sentences are joined to the next one."),
        BI.SettingField("speak_drop_unfinished","Drop an Unfinished Last Sentence (speak while writing)","checkbox",default=True,advanced=True),
        BI.SettingField("kv_cache_type","KV Cache Type (llama.cpp connections)","select",default="",options=[("","(connection default)")]+[(k,k) for k in ("f16","q8_0","q5_1","q5_0","q4_1","q4_0","iq4_nl")],advanced=True),
        BI.SettingField("cache_session","Saved-Context Session","text",advanced=True,hint="Nodes sharing a session reuse each other's prefilled context. Blank = shared default."),
        *_pool_fields()[:2], BI.SettingField("conn_id", "Connection", type="select", default="", options=_generate_conn_options, hint="Follows Modality: the chat model's connection (text), the text encoder (image), the video node (video). Blank = pool-resolved."), _pool_fields()[3], _key_map_field()], guide="One universal generation node - text, image or video, picked by Modality. With no connection pinned, resolves one from this node's resource pool (pipeline pool intersected with this node's own tags) using Priority. Video picks by capability from the frames given (start + end, start, none); the pool may fall back to a lesser capability, a pinned connection may not.")

    register_node_type("transform", node_transform, "Transform (deterministic)", in_keys=["input"], out_keys=["value"], config_schema=[
        BI.SettingField("mode", "Mode", "select", default="template", options=[("template","Template fill"), ("expr","Python expression"), ("chunk","Chunk (paragraph-safe split)"), ("regex_find","Regex find"), ("regex_replace","Regex replace"),("python","Inline script")]),
        BI.SettingField("template","Template","textarea",default="{input}"),
        BI.SettingField("expr","Expression","text",default="input",advanced=True),
        BI.SettingField("vars_json","Variables (JSON)","json",default={},advanced=True),
        BI.SettingField("pattern","Regex Pattern","text",advanced=True),
        BI.SettingField("dotall","Regex DOTALL","checkbox",default=False,advanced=True),
        BI.SettingField("replacement_template","Replacement Template","text",advanced=True),
        BI.SettingField("replace_mode","Replace","select",default="all",options=[("all","All matches"),("first","First match")],advanced=True),
        BI.SettingField("script_body","Inline Script Body","textarea",advanced=True),
        BI.SettingField("script_path","Script File (instead of inline)","file_picker",advanced=True),
        BI.SettingField("timeout_s","Timeout (s)","number",default=600,step=1,advanced=True),
        BI.SettingField("chunk_chars","Chunk Target Size (chars)","number",default=4000,step=1,advanced=True),
        BI.SettingField("overlap_chars","Chunk Overlap (chars, carried into next chunk)","number",default=0,step=1,advanced=True),
        BI.SettingField("hard_split_ratio", "Hard-split threshold (x target size)", "number", default=2.0, advanced=True),
        _key_map_field()], guide="One deterministic node covering template-fill, a sandboxed Python expression, regex search/replace, or a full inline/file script. No AI call, no resource pool.")

    register_node_type("file_read", node_file_read, "File Read", in_keys=["path"], out_keys=["text"], config_schema=[
        BI.SettingField("path","Path Override","text",advanced=True, hint="Leave blank to read the resolved 'path' in-key instead."),
        BI.SettingField("fm_root","FS Root","file_picker",default="./data/_common"),
        _key_map_field()], guide="Reads a file's text content.")

    register_node_type("file_write", node_file_write, "File Write (shadow-staged)", in_keys=["path","content"], out_keys=["path","status"], config_schema=[
        BI.SettingField("path","Destination Path Template","text", hint="Supports {key} against real pipeline keys, e.g. articles/{slug}.md."),
        BI.SettingField("fm_root","FS Root","file_picker",default="./data/_common"),
        BI.SettingField("binary","Binary Content","checkbox",default=False,advanced=True, hint="Content in-key holds raw bytes or a source file path rather than text."),
        BI.SettingField("allow_empty","Allow Empty Content","checkbox",default=False,advanced=True),
        BI.SettingField("auto_accept","Auto-accept (skip review)","checkbox",default=False,advanced=True,hint="Writes and immediately accepts into the real file, with full rollback history still kept. Use only for files this pipeline itself owns and manages - never for the user's own source content."),
        BI.SettingField("source_root","Source Root (binary content, when content is a filename)","file_picker",default=".",advanced=True),
        _key_map_field()], guide="Never writes directly - always through Shadow Stage (accept/reject before it touches the real file).")

    register_node_type("file_list", node_file_list, "File List", in_keys=[], out_keys=["files"], config_schema=[
        BI.SettingField("root","Target Directory","text",default="./data/ai_tools/_knowledge"),
        BI.SettingField("extensions","Extensions (comma-sep)","text", advanced=True),
        _key_map_field()], guide="Lists relative file paths under a directory. No dependencies - runs as soon as the pipeline starts.")

    register_node_type("knowledge", node_knowledge, "Knowledge (query / insert / entities)", in_keys=["input"], out_keys=["response","raw"], config_schema=[
        BI.SettingField("mode","Mode","select",default="query",options=[("query","Query"),("insert","Insert Text"),("entities","List Entities")]),
        BI.SettingField("query_template","Query Template","textarea",default="{input}"),
        BI.SettingField("query_mode","Search Mode","select",default="hybrid",options=[("hybrid","Hybrid"),("local","Local"),("global","Global"),("naive","Naive"),("mix","Mix")]),
        BI.SettingField("text_template","Insert Text Template","textarea",advanced=True),
        BI.SettingField("source_label","Insert Source Label","text",advanced=True),
        BI.SettingField("limit","Entity Limit","number", default=500,step=1,advanced=True),
        *_pool_fields(include_model=False, conn_type="knowledge_query"), _key_map_field()], guide="One node for the three LightRAG operations. Resource pool resolves a lightrag connection the same way Generate resolves an LLM connection.")

    register_node_type("pipeline", node_pipeline, "Call Pipeline", in_keys=[], out_keys=[], config_schema=[
        BI.SettingField("pipeline_id","Pipeline","select", options=_pipeline_options),
        BI.SettingField("import_keys","Import Keys (comma-sep, actual names)","text", hint="Which of THIS pipeline's keys to seed the sub-pipeline with, under the same names. Also add these to Extra In Keys below so the scheduler waits for them."),
        BI.SettingField("export_keys","Export Keys (comma-sep, actual names)","text"),
        _key_map_field()], guide="Runs another saved pipeline to completion inline. import_keys/export_keys are the only channel between parent and sub-pipeline data objects.")

    register_node_type("pipeline_foreach", node_pipeline_foreach, "For Each Item, Call Pipeline", in_keys=["items"], out_keys=["results","count"], config_schema=[
        BI.SettingField("pipeline_id","Pipeline","select", options=_pipeline_options),
        BI.SettingField("item_key","Item Key (sub-pipeline name for one item)","text",default="item"),
        BI.SettingField("import_keys","Import Keys (comma-sep, actual names)","text",advanced=True),
        BI.SettingField("export_keys","Export Keys (comma-sep, actual names)","text"),
        BI.SettingField("on_item_error","When an Item's Run Fails","select",default="continue",options=[("continue","Continue (its result carries _error)"),("stop","Stop (this node fails)")],advanced=True),
        _key_map_field()], guide="Runs the sub-pipeline once per item in 'items', sequentially, collecting each run's export_keys into 'results'. A failed item's result carries '_error'.")

    register_node_type("pipeline_reduce", node_pipeline_reduce, "For Each Item, Call Pipeline (Sequential Fold)", in_keys=["items","accumulator"], out_keys=["accumulator"], config_schema=[
        BI.SettingField("pipeline_id","Pipeline","select", options=_pipeline_options),
        BI.SettingField("item_key","Item Key (sub-pipeline name for one item)","text",default="item"),
        BI.SettingField("accumulator_key","Accumulator Key (sub-pipeline name for the running value)","text",default="accumulator"),
        BI.SettingField("accumulator_export_key","Sub-pipeline's Export Key for the Updated Accumulator","text",default="",advanced=True,hint="Leave blank to reuse Accumulator Key."),
        BI.SettingField("import_keys","Import Keys (comma-sep, actual names, constant across every iteration)","text",advanced=True),
        _key_map_field()], guide="Like For Each Item but SEQUENTIAL: each call sees the previous call's updated accumulator, so the sub-pipeline builds on its own output as it works through the list in order. Use For Each instead when items are independent.")

    register_node_type("branch", node_branch, "Branch (decide + call one pipeline)", in_keys=["input"], out_keys=["decision"], config_schema=[
        BI.SettingField("decide_mode","Decide Using","select",default="expr",options=[("expr","Python expression"),("llm","LLM (enforced options)")]),
        BI.SettingField("decide_expr","Decision Expression","text",default="input",advanced=True),
        BI.SettingField("decide_template","LLM Decision Prompt","textarea",advanced=True),
        BI.SettingField("options","LLM Options (comma-sep)","text",advanced=True),
        BI.SettingField("routes_json","Routes (JSON: value -> pipeline id)","json",default={}),
        BI.SettingField("default_pipeline_id","Default Pipeline ID","select",options=_pipeline_options,advanced=True),
        BI.SettingField("import_keys","Import Keys (comma-sep)","text"),
        BI.SettingField("export_keys","Export Keys (comma-sep)","text"),
        BI.SettingField("vars_json","Variables (JSON: name -> template)","json",default={},advanced=True),
        *_pool_fields(), _key_map_field()], guide="Decides a value, looks it up in Routes, calls whichever sub-pipeline matches (falling back to Default Pipeline ID). Only the chosen path actually runs.")

    register_node_type("transcribe", node_transcribe, "Transcribe (speech to text)", in_keys=["audio_b64"], out_keys=["text","segments","language","dropped_text"], config_schema=[
        BI.SettingField("language","Language (ISO code, blank = auto-detect)","text"),
        BI.SettingField("vad_filter","Skip silence (VAD)","checkbox",default=True),
        BI.SettingField("skip_empty","Drop Nothing-Said Audio","checkbox",default=True,hint="An empty or punctuation-only transcript gives no 'text' (only 'dropped_text'), so the nodes waiting on 'text' never run."),
        BI.SettingField("ignore_phrases","Ignore These Transcripts (one per line)","textarea",advanced=True,hint="Whole transcripts treated as nothing said - the recognizer's noise inventions, e.g. Thank you. / Thanks for watching!"),
        BI.SettingField("max_no_speech_prob","Drop Segments Rated Not-Speech Above","number",default=None,advanced=True,hint="0-1, the recognizer's own per-segment no_speech_prob (speech_direct reports it). Blank = keep every segment; 0.6 is a reasonable start."),
        BI.SettingField("initial_prompt","Vocabulary Prompt","textarea",advanced=True,hint="Names and jargon the recognizer should expect. Supports {key}."),
        BI.SettingField("stt_model","Recognizer Model Override","text",advanced=True,hint="Blank = the node's default (e.g. small, medium, large-v3)."),
        *_pool_fields(include_model=False, conn_type="speech_to_text"), _key_map_field()], guide="Sends base64 audio (webm/ogg/mp4/wav/mp3) in 'audio_b64' to any connection declaring speech_to_text and returns the text. Pair with Speak and Generate for a voice conversation.")

    register_node_type("identify_speaker", node_identify_speaker, "Identify Speaker (voiceprint)", in_keys=["audio_b64", "profiles"], out_keys=["speaker", "speaker_code", "speaker_role", "speaker_score", "speaker_tag"], config_schema=[
        BI.SettingField("match_threshold","Same Person At Or Above (cosine 0-1)","number",default=0.5,hint="WeSpeaker ResNet34: the same voice usually scores 0.6-0.8, different voices under 0.3. Raise if people get mixed up."),
        BI.SettingField("guess_threshold","Likely Match At Or Above","number",default=0.35,hint="Between this and the match threshold the best profile is used with the guess mark. Below it the speaker is the Other name."),
        BI.SettingField("tag_format","Tag Format","text",default="({name}{guess}): ",hint="Fields: {name} {code} {role} {guess}. '({code}): ' gives number signatures instead of names."),
        BI.SettingField("guess_mark","Guess Mark","text",default="?",advanced=True),
        BI.SettingField("other_name","Name for Unknown Voices","text",default="Other",advanced=True),
        BI.SettingField("other_code","Code for Unknown Voices","text",default="0000",advanced=True),
        *_pool_fields(include_model=False, conn_type="speaker_embed"), _key_map_field()], guide="Tells enrolled voices apart: 'profiles' is a list of {name, code, role, embedding}; the result labels the utterance and carries the profile's role (e.g. owner / trusted / guest) for later permission checks.")

    register_node_type("call_capability", node_call_capability, "Call Capability (any connection)", in_keys=[], out_keys=["result"], config_schema=[
        BI.SettingField("capability","Capability","text",hint="One the connection's profile declares, e.g. video_join, model_list, voice_list."),
        BI.SettingField("payload_json","Payload (JSON; values may be {key} templates)","json",default={},hint="A value that is exactly one {key} passes that key's raw value (a list stays a list). Add the keys it uses to Extra In Keys so the node waits for them."),
        BI.SettingField("out_fields","Response Fields to Output (comma-sep)","text",hint="Each becomes an output key of the same name (declare them in Extra Out Keys for the builder view); the whole response is always in 'result'."),
        BI.SettingField("timeout_s","Timeout (s, blank = the connection's)","number",default=None,step=1,advanced=True),
        *_pool_fields(include_model=False)[:2], BI.SettingField("conn_id","Connection","select",default="",options=_all_conn_options,hint="Blank = the pool picks a connection declaring the capability."), _key_map_field()], guide="Calls one capability any connection declares and returns its JSON answer. New node features become usable in pipelines without a dedicated node type.")

    register_node_type("speak", node_speak, "Speak (text to speech)", in_keys=["text"], out_keys=["audio_b64","format","duration_s"], config_schema=[
        BI.SettingField("voice","Voice","text",hint="Voice id on the speech node (e.g. en_US-lessac-medium). Blank = the node's default."),
        BI.SettingField("length_scale","Speaking Pace (length scale)","number",default=None,hint=">1 slower, <1 faster. Blank = the voice's own pace."),
        *_pool_fields(include_model=False, conn_type="text_to_speech"), _key_map_field()], guide="Synthesizes 'text' through any connection declaring text_to_speech and returns base64 audio. The speech node strips thinking blocks, code and markdown before speaking.")
