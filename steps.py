"""steps.py - universal node type registry + implementations for ai_manager pipelines.
Every node type is one async function: fn(config: dict, data: dict, ctx: NodeContext) -> dict.
It reads whatever it needs from the shared pipeline `data` object - ctx.get()/ctx.resolve() are convenience helpers over the node's own key_map, not an access restriction;
the in_keys/out_keys a type declares are a scheduling contract, not a firewall - and returns a dict of LOGICAL output names -> values.
The engine remaps those to actual pipeline keys via the node's key_map and merges them into `data`.
"""
import re, json, sys, asyncio, uuid, time
from pathlib import Path
from tools.ai_manager import engine, resources
from tools.ai_manager.connections import get_conn, lightrag_query, lightrag_insert_text, lightrag_list_entities, stream_llm, flux2_encode, flux2_generate, list_models_sync

_NODE_TYPES: dict = {}
ENV: dict = {}
def init(env: dict):
    global ENV
    ENV = env

def register_node_type(name, fn, label="", in_keys=None, out_keys=None, config_schema=None, guide=""):
    _NODE_TYPES[name] = {"fn": fn, "label": label or name, "in_keys": in_keys or [], "out_keys": out_keys or [], "config_schema": config_schema or [], "guide": guide}
def get_node_type(name: str) -> dict: return _NODE_TYPES.get(name)
def list_node_types() -> list: return [{"type": k, **{kk: vv for kk, vv in v.items() if kk != "fn"}} for k, v in _NODE_TYPES.items()]

class NodeContext:
    """Wraps one node's own key_map so a type's implementation only ever deals in its own logical names, never in actual pipeline key strings
    - the same function works unmodified no matter what keys a pipeline instance wires it to."""
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

    async def progress(self, message): await ENV["push_to_client"](self.username, {"t":"pipeline_event","job_id":self.job_id,"event":"running","payload":{"node":self.node.id,"message":message}})
    async def stream(self, key, delta): await ENV["push_to_client"](self.username, {"t":"pipeline_stream","job_id":self.job_id,"key":key,"delta":delta})

# --- generate: any AI generation call - text or image, picked by modality ---

async def node_generate(config: dict, data: dict, ctx: NodeContext) -> dict:
    modality = config.get("modality", "text")
    priority = config.get("priority") or ctx.pool_cfg.get("priority", "balanced")
    tags = [t.strip() for t in str(config.get("cnode_tags","")).split(",") if t.strip()]
    t0 = time.time()
    if modality == "image":
        enc_conn, cnode = (get_conn(config.get("conn_id","")), None) if config.get("conn_id") else (None, None)
        if not enc_conn:
            picked = resources.pick_conn(resources.resolve_candidates(ctx.pool_cfg, tags, "flux2_text"), "flux2_text", priority)
            if not picked: raise RuntimeError("generate(image): no flux2_text connection matches this node's resource pool")
            cnode, enc_conn = picked
        picked2 = resources.pick_conn(resources.resolve_candidates(ctx.pool_cfg, tags, "flux2_image"), "flux2_image", priority)
        if not picked2: raise RuntimeError("generate(image): no flux2_image connection matches this node's resource pool")
        _, img_conn = picked2
        prompt = ctx.resolve(config.get("user_template") or "{input}")
        job_tag = f"{ctx.job_id}_{uuid.uuid4().hex[:6]}"
        enc = await flux2_encode(enc_conn, prompt, job_id=job_tag, max_sequence_length=config.get("max_sequence_length", 512))
        if enc.get("error"): raise RuntimeError(f"generate(image) encode: {enc['error']}")
        gen = await flux2_generate(img_conn, {"prompt": prompt, "embed_job_id": job_tag, "width": config.get("width",1024), "height": config.get("height",1024), "steps": config.get("steps",4), "guidance_scale": config.get("cfg",1.0), "shift": config.get("shift",1.0), "seed": config.get("seed",-1)})
        if gen.get("error"): raise RuntimeError(f"generate(image): {gen['error']}")
        if cnode: resources.log_usage(cnode["id"], "generate:image", time.time()-t0)
        return {"file_name": gen["file_name"]}
    conn, cnode = (get_conn(config.get("conn_id","")), None) if config.get("conn_id") else (None, None)
    if not conn:
        picked = resources.pick_conn(resources.resolve_candidates(ctx.pool_cfg, tags, "ollama"), "ollama", priority)
        if not picked: raise RuntimeError("generate: no connection matches this node's resource pool (check pipeline/node whitelist-blacklist tags)")
        cnode, conn = picked
    model = config.get("model","")
    if not model:
        models = list_models_sync(conn)
        if not models: raise RuntimeError(f"generate: connection {conn.get('_id','')} has no models available")
        model = models[0]
    seed_kwargs = {} if config.get("seed") in (None,"",-1) else {"seed": int(config["seed"])}
    raw_opts = config.get("enforce_options","")
    options = [o.strip() for o in raw_opts.split(",") if o.strip()] if isinstance(raw_opts,str) else (raw_opts or [])
    sys_p = ctx.resolve(config.get("system_prompt") or "")
    if options: sys_p = (sys_p + f"\n\nRespond with exactly one of these words and nothing else: {', '.join(options)}").strip()
    messages = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content": ctx.resolve(config.get("user_template") or "{input}")}]
    full, thinking_full = "", ""
    async for text, thinking in stream_llm(conn, messages, model, think=config.get("think", False), temperature=config.get("temperature", 0.7), num_ctx=config.get("num_ctx", 16384), num_predict=config.get("num_predict", -1), **seed_kwargs):
        full += text; thinking_full += thinking
        await ctx.stream("text", text)
    if cnode: resources.log_usage(cnode["id"], "generate:text", time.time()-t0)
    result = {"text": full, "thinking": thinking_full}
    if options: result["choice"] = next((o for o in options if o.lower() in full.lower()), options[0])
    return result

# --- transform: deterministic data shaping, no AI call, no resource pool ---

async def node_transform(config: dict, data: dict, ctx: NodeContext) -> dict:
    mode = config.get("mode", "template")
    if mode == "template": return {"value": ctx.resolve(config.get("template", "{input}"))}
    if mode == "expr":
        try: var_templates = json.loads(config.get("vars_json","{}") or "{}")
        except Exception: var_templates = {}
        local_vars = {name: ctx.resolve(tpl) for name, tpl in var_templates.items()}
        safe_builtins = {"len":len,"str":str,"int":int,"float":float,"min":min,"max":max,"sorted":sorted,"round":round}
        try: value = eval(config.get("expr","input"), {"__builtins__": safe_builtins}, {**local_vars, "input": ctx.get("input")})
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
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(fm_root) / "_shadow"))
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
        picked = resources.pick_conn(resources.resolve_candidates(ctx.pool_cfg, tags, "lightrag"), "lightrag", priority)
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
    sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1)
    return {k: sub_data.get(k, "") for k in export_keys}

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
    results = []
    for i, item in enumerate(items):
        await ctx.progress(f"item {i+1}/{len(items)}")
        sub_inputs = {item_key: item, **{k: data.get(k, "") for k in import_keys}}
        sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1)
        results.append({k: sub_data.get(k, "") for k in export_keys})
    return {"results": results, "count": len(results)}

async def node_branch(config: dict, data: dict, ctx: NodeContext) -> dict:
    """Decides a value (expression or LLM), looks it up in Routes, and calls whichever sub-pipeline matches.
    The branch is one atomic node to the outer pipeline's scheduler - only the chosen sub-pipeline actually runs;
    the others are simply never invoked, exactly like any other unreached path in this architecture."""
    if config.get("decide_mode", "expr") == "llm":
        decision = (await node_generate({**config, "enforce_options": config.get("options",""), "user_template": config.get("decide_template","{input}")}, data, ctx)).get("choice","")
    else:
        try: decision = str(eval(config.get("decide_expr","input"), {"__builtins__": {"len":len,"str":str,"int":int}}, {"input": ctx.get("input")}))
        except Exception as e: raise RuntimeError(f"branch: decide_expr failed: {e}")
    try: routes = json.loads(config.get("routes_json","{}") or "{}")
    except Exception: routes = {}
    pid = routes.get(decision) or config.get("default_pipeline_id","")
    if not pid: raise RuntimeError(f"branch: no route for decision '{decision}' and no default_pipeline_id set")
    import_keys = [k.strip() for k in str(config.get("import_keys","")).split(",") if k.strip()]
    export_keys = [k.strip() for k in str(config.get("export_keys","")).split(",") if k.strip()]
    sub_inputs = {k: data.get(k, "") for k in import_keys}
    sub_data = await engine.run_inline(ctx.username, pid, inputs=sub_inputs, pool_cfg=ctx.pool_cfg, depth=ctx.depth+1)
    return {**{k: sub_data.get(k, "") for k in export_keys}, "decision": decision}

# --- registration ---

def register_builtins():
    BI = ENV["tools"]["built_ins"]
    def _key_map_field(): return BI.SettingField("key_map", "Key Map (JSON: logical -> actual pipeline key)", type="json", default={}, advanced=True, hint='Only needed to rename this node\'s in/out keys, e.g. {"input":"user_query","text":"draft"}. Unmapped logical names pass through unchanged.')
    def _pool_fields(): return [BI.SettingField("cnode_tags", "Resource Pool Tags (comma-sep)", type="text", advanced=True, hint="Narrows the pipeline's own pool to CNodes carrying ALL these tags. Blank = use the whole pipeline pool."),
                                 BI.SettingField("priority", "Priority", type="select", default="", options=[("","(inherit pipeline default)"),("speed","Speed"),("balanced","Balanced"),("quality","Quality")], advanced=True),
                                 BI.SettingField("conn_id", "Pin Connection (optional)", type="text", advanced=True, hint="Explicit connection id - bypasses the resource pool entirely when set.")]

    register_node_type("generate", node_generate, "Generate (text or image)", in_keys=["input"], out_keys=["text","thinking"], config_schema=[
        BI.SettingField("modality","Modality","select",default="text", options=[("text","Text"),("image","Image")]),
        BI.SettingField("model","Model (blank = first available)","text",default=""),
        BI.SettingField("system_prompt","System Prompt","textarea",default="You are a helpful AI assistant."),
        BI.SettingField("user_template","User/Prompt Template","textarea",default="{input}", hint="{key} substitutes real pipeline keys directly."),
        BI.SettingField("temperature","Temperature","number",default=0.7),
        BI.SettingField("num_ctx","Context Window","number",default=16384,step=1),
        BI.SettingField("num_predict","Max Output Tokens","number",default=-1,step=1),
        BI.SettingField("think","Enable Thinking Mode","checkbox",default=False),
        BI.SettingField("enforce_options","Enforced Options (comma-sep)","text",advanced=True),
        BI.SettingField("seed","Seed (blank/-1 = random)","number",default=None,advanced=True,step=1),
        BI.SettingField("width","Width (image)","number",default=1024,step=16,advanced=True),
        BI.SettingField("height","Height (image)","number",default=1024,step=16,advanced=True),
        BI.SettingField("steps","Steps (image)","number",default=4,step=1,advanced=True),
        BI.SettingField("cfg","Guidance Scale (image)","number",default=1.0,advanced=True),
        *_pool_fields(), _key_map_field()],
        guide="One universal generation node - text or image, picked by Modality. With no connection pinned, resolves one from this node's resource pool (pipeline pool intersected with this node's own tags) using Priority.")

    register_node_type("transform", node_transform, "Transform (deterministic)", in_keys=["input"], out_keys=["value"], config_schema=[
        BI.SettingField("mode","Mode","select",default="template", options=[("template","Template fill"),("expr","Python expression"),("regex_find","Regex find"),("regex_replace","Regex replace"),("python","Inline script")]),
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
        _key_map_field()],
        guide="One deterministic node covering template-fill, a sandboxed Python expression, regex search/replace, or a full inline/file script. No AI call, no resource pool.")

    register_node_type("file_read", node_file_read, "File Read", in_keys=["path"], out_keys=["text"], config_schema=[
        BI.SettingField("path","Path Override","text",advanced=True, hint="Leave blank to read the resolved 'path' in-key instead."),
        BI.SettingField("fm_root","FS Root","file_picker",default="./data/_common"), _key_map_field()],
        guide="Reads a file's text content.")

    register_node_type("file_write", node_file_write, "File Write (shadow-staged)", in_keys=["path","content"], out_keys=["path","status"], config_schema=[
        BI.SettingField("path","Destination Path Template","text", hint="Supports {key} against real pipeline keys, e.g. articles/{slug}.md."),
        BI.SettingField("fm_root","FS Root","file_picker",default="./data/_common"),
        BI.SettingField("binary","Binary Content","checkbox",default=False,advanced=True, hint="Content in-key holds raw bytes or a source file path rather than text."),
        BI.SettingField("allow_empty","Allow Empty Content","checkbox",default=False,advanced=True),
        _key_map_field()],
        guide="Never writes directly - always through Shadow Stage (accept/reject before it touches the real file).")

    register_node_type("file_list", node_file_list, "File List", in_keys=[], out_keys=["files"], config_schema=[
        BI.SettingField("root","Target Directory","text",default="./data/ai_tools/_knowledge"),
        BI.SettingField("extensions","Extensions (comma-sep)","text",advanced=True), _key_map_field()],
        guide="Lists relative file paths under a directory. No dependencies - runs as soon as the pipeline starts.")

    register_node_type("knowledge", node_knowledge, "Knowledge (query / insert / entities)", in_keys=["input"], out_keys=["response","raw"], config_schema=[
        BI.SettingField("mode","Mode","select",default="query",options=[("query","Query"),("insert","Insert Text"),("entities","List Entities")]),
        BI.SettingField("query_template","Query Template","textarea",default="{input}"),
        BI.SettingField("query_mode","Search Mode","select",default="hybrid",options=[("hybrid","Hybrid"),("local","Local"),("global","Global"),("naive","Naive"),("mix","Mix")]),
        BI.SettingField("text_template","Insert Text Template","textarea",advanced=True),
        BI.SettingField("source_label","Insert Source Label","text",advanced=True),
        BI.SettingField("limit","Entity Limit","number",default=500,step=1,advanced=True),
        *_pool_fields(), _key_map_field()],
        guide="One node for the three LightRAG operations. Resource pool resolves a lightrag connection the same way Generate resolves an LLM connection.")

    register_node_type("pipeline", node_pipeline, "Call Pipeline", in_keys=[], out_keys=[], config_schema=[
        BI.SettingField("pipeline_id","Pipeline","select", options=_pipeline_options),
        BI.SettingField("import_keys","Import Keys (comma-sep, actual names)","text", hint="Which of THIS pipeline's keys to seed the sub-pipeline with, under the same names. Also add these to Extra In Keys below so the scheduler waits for them."),
        BI.SettingField("export_keys","Export Keys (comma-sep, actual names)","text"),
        _key_map_field()],
        guide="Runs another saved pipeline to completion inline. import_keys/export_keys are the only channel between parent and sub-pipeline data objects.")

    register_node_type("pipeline_foreach", node_pipeline_foreach, "For Each Item, Call Pipeline", in_keys=["items"], out_keys=["results","count"], config_schema=[
        BI.SettingField("pipeline_id","Pipeline","select", options=_pipeline_options),
        BI.SettingField("item_key","Item Key (sub-pipeline name for one item)","text",default="item"),
        BI.SettingField("import_keys","Import Keys (comma-sep, actual names)","text",advanced=True),
        BI.SettingField("export_keys","Export Keys (comma-sep, actual names)","text"), _key_map_field()],
        guide="Runs the sub-pipeline once per item in 'items', sequentially, collecting each run's export_keys into 'results'.")

    register_node_type("branch", node_branch, "Branch (decide + call one pipeline)", in_keys=["input"], out_keys=["decision"], config_schema=[
        BI.SettingField("decide_mode","Decide Using","select",default="expr",options=[("expr","Python expression"),("llm","LLM (enforced options)")]),
        BI.SettingField("decide_expr","Decision Expression","text",default="input",advanced=True),
        BI.SettingField("decide_template","LLM Decision Prompt","textarea",advanced=True),
        BI.SettingField("options","LLM Options (comma-sep)","text",advanced=True),
        BI.SettingField("routes_json","Routes (JSON: value -> pipeline id)","json",default={}),
        BI.SettingField("default_pipeline_id","Default Pipeline ID","text",advanced=True),
        BI.SettingField("import_keys","Import Keys (comma-sep)","text"),
        BI.SettingField("export_keys","Export Keys (comma-sep)","text"),
        *_pool_fields(), _key_map_field()],
        guide="Decides a value, looks it up in Routes, calls whichever sub-pipeline matches (falling back to Default Pipeline ID). Only the chosen path actually runs.")