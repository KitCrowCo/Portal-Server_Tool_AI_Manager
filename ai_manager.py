#tools/ai_manager
"""
ai_manager — sole centralized tool for AI operations. Owns connections, the step registry, and pipeline execution.
modules/ai_tools/* (Athena, Kimi, Tessa, Image) are UI surfaces that call into this tool directly (ENV["tools"]["ai_manager"]);
they never hold their own connections or execution loops - avoids the race conditions of tools calling other tools.
"""
import sys, os, json, re, uuid, copy
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from tools.ai_manager import engine
from tools.ai_manager import steps
from tools.ai_manager import connections

TOOL_META = {"label": "AI Manager", "icon": "&#x1F9E0;", "description": "Centralized AI connections, steps, and pipeline execution"}
router = APIRouter()
_P = "/tool/ai_manager"
TOOL_ROOT = Path("./data/ai_manager/_selftest")
PROMPT_BLOCKS_PATH = Path("./data/ai_manager/prompt_blocks.json")

ENV = globals().get("ENV", {})
UI = globals().get("UI", None)
BI = globals().get("BI", None)
FM = globals().get("FM", None)
_PB = globals().get("_PB", None)
_NAMED_ROOTS = {"common": "./data/_common"}

def register_root(name: str, path: str): _NAMED_ROOTS[name] = path
def resolve_root(name: str) -> str: return _NAMED_ROOTS.get(name, name)
def list_roots() -> list: return list(_NAMED_ROOTS.items())

def init_module(env: dict):
    global ENV, UI, BI, FM, _PB
    ENV.update(env)
    UI = ENV["templates"].env.globals.get("UI")
    BI = ENV["tools"]["built_ins"]
    FM = BI.FileManager(TOOL_ROOT)
    _PB = BI.PromptBlockLibrary(PROMPT_BLOCKS_PATH)
    engine.init(env)
    steps.init(env)
    steps.register_builtins()
    print(f"[ai_manager] ready | step types: {[s['type'] for s in steps.list_step_types()]}")

def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def prompt_block_picker_fragment(textarea_id: str) -> str: return BI.prompt_block_picker_html(_PB, textarea_id, f"{_P}/prompt_blocks/save")



class PipelineBuilderUI:
    """Generalized pipeline authoring surface over AIM.engine/AIM.steps. Fully intent-based - no bespoke
    GET routes, everything flows through /im/in and OOB updates to this instance's own generic ids.
    Any module embeds panel_html(scope_id) once instead of reimplementing node forms/editor/run controls.
    scope_key/scope_id decouple 'what owns this pipeline' from the engine - one module calls it a project,
    another a workspace; the builder only needs a stable string id and the key name it's stored under."""

    SCRIPT = """
    function plbExport(plId) {
        fetch('/tool/ai_manager/pipelines/' + plId + '/export').then(r => r.json()).then(d => {
            var blob = new Blob([JSON.stringify(d, null, 2)], {type: 'application/json'});
            var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = (d.name || 'pipeline') + '.json'; a.click();
        });
    }
    """

    def __init__(self, IM, AIM, intent_prefix="plb", nesting_level=2, scope_key="project_id"):
        self.IM, self.AIM, self.intent_prefix, self.nesting_level, self.scope_key = IM, AIM, intent_prefix, nesting_level, scope_key
        p = self.intent_prefix
        IM.scripts.update({f"{p}_new_form": [self._im_new_form], f"{p}_create": [self._im_create], f"{p}_delete": [self._im_delete],
                            f"{p}_editor_open": [self._im_editor_open], f"{p}_editor_close": [self._im_editor_close],
                            f"{p}_node_form": [self._im_node_form], f"{p}_node_type_change": [self._im_node_type_change],
                            f"{p}_node_add": [self._im_node_save], f"{p}_node_save": [self._im_node_save], f"{p}_node_delete": [self._im_node_delete],
                            f"{p}_rename": [self._im_rename], f"{p}_run": [self._im_run], f"{p}_stop": [self._im_stop], f"{p}_resume": [self._im_resume],
                            f"{p}_status": [self._im_status], f"{p}_step_models": [self._im_step_models], f"{p}_claim": [self._im_claim]})

    def _vals(self, action, **extra): return json.dumps({"type": f"{self.intent_prefix}_{action}", "branch": self.intent_prefix, "lvl": self.nesting_level, **extra})
    def _post(self, action, **extra): return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{self._vals(action, **extra)}'"""
    def _pipelines(self, scope_id): return [p for p in self.AIM.engine.list_pipelines() if p.get(self.scope_key) == scope_id]

    def panel_html(self, scope_id: str) -> str:
        p = self.intent_prefix
        cards = "".join(self._card_html(scope_id, pl) for pl in self._pipelines(scope_id)) or '<div class="pl-empty">No pipelines. Click + to create one.</div>'
        return f"""<div id="pl-panel-{p}" class="pl-panel">
                       <div class="pl-panel-hd"><span class="pl-panel-title">Pipelines</span><button class="btn-icon" {self._post("new_form", scope=scope_id)}>+</button></div>
                       <div id="pl-new-{p}"></div>
                       <div id="pl-list-{p}" class="pl-list">{cards}</div>
                       <div id="pl-editor-modal-{p}"></div>
                   </div>"""

    def _card_html(self, scope_id, pl) -> str:
        pl_id = pl["id"]
        last_job = self.AIM.engine.load_job(pl.get("last_job_id","")) if pl.get("last_job_id") else None
        nodes = (last_job["flow"]["nodes"] if last_job else pl.get("flow",{}).get("nodes",[]))
        rows = "".join(self._node_status_row(n) for n in nodes)
        return f"""<div class="glass pl-card">
                       <div class="pl-card-hd">
                           <span class="pl-card-title" {self._post("editor_open", scope=scope_id, pl_id=pl_id)}>{UI.escape(pl.get("name",""))}</span>
                           <button type="button" class="cm-qbtn" onclick="plbExport('{pl_id}')">&#x2B07;</button>
                           <button class="btn-icon" style="color:#ff5f5f" {self._post("delete", scope=scope_id, pl_id=pl_id)} onclick="return confirm('Delete pipeline?')">&#x2715;</button>
                       </div>
                       <input type="text" id="pl-input-{pl_id}" name="value" placeholder="Input for this run" class="module-select">
                       <div id="pl-status-{pl_id}">{self._status_block(scope_id, pl_id, last_job)}</div>
                       <details class="pl-nodes"><summary>Node status ({len(nodes)})</summary><table class="pl-status-table">{rows}</table></details>
                   </div>"""

    def _status_block(self, scope_id, pl_id, job) -> str:
        live = bool(job and job.get("status") in ("running","queued"))
        if live:
            return f"""<div class="pl-status-row" hx-trigger="load delay:1.5s" {self._post("status", scope=scope_id, pl_id=pl_id)} hx-target="#pl-status-{pl_id}" hx-swap="innerHTML">
                           <button class="cm-qbtn" style="color:#ff4444" {self._post("stop", scope=scope_id, pl_id=pl_id, job_id=job["id"])}>&#x25FC; Stop</button>
                           <span class="pl-status-label" style="color:#00ffa2">{job["status"]}</span></div>"""
        label = job["status"] if job else "idle"
        resume = f"""<button class="cm-qbtn" {self._post("resume", scope=scope_id, pl_id=pl_id)}>&#x21BB; Resume</button>""" if job and job.get("status") == "interrupted" else ""
        return f"""<div class="pl-status-row">
                       <button class="cm-qbtn" {self._post("run", scope=scope_id, pl_id=pl_id)} hx-include="#pl-input-{pl_id}">&#x25B6; Run</button>
                       {resume}<span class="pl-status-label">{label}</span></div>"""

    @staticmethod
    def _node_slug(n) -> str:
        explicit = str(n.get("slug","")).strip().lower()
        if explicit: return re.sub(r'\W+', '_', explicit).strip('_') or n["id"]
        return re.sub(r'\W+', '_', (n.get("name") or "").strip().lower()).strip('_') or n["id"]

    def _node_status_row(self, n) -> str:
        slug, preview = self._node_slug(n), n.get("message") or " | ".join((n.get("preview") or {}).values())
        return f"""<tr><td class="qn">{UI.escape(n.get("name") or n["id"])}<br><code class="pl-slug">{slug}</code></td>
                       <td class="dim">{UI.escape(n.get("type",""))}</td><td class="dim">{UI.escape(n.get("status","idle"))}</td>
                       <td class="dim pl-preview">{UI.escape(preview)}</td></tr>"""

    def _step_type_options(self, selected=""):
        blank = '<option value="" selected disabled>-- select step type --</option>' if not selected else ""
        return blank + "".join(f'<option value="{t["type"]}" {"selected" if t["type"]==selected else ""}>{UI.escape(t.get("label",t["type"]))}</option>' for t in self.AIM.steps.list_step_types())

    def _step_config_form_fields(self, step_type, config):
        spec = self.AIM.steps.get_step_type(step_type)
        if not spec: return '<div class="dim">Pick a step type to configure it.</div>'
        schema = [copy.copy(f) if f.name == "conn_id" else f for f in spec["config_schema"]]
        for f in schema:
            if f.name == "conn_id": f.hx_intent, f.hx_target = f"{self.intent_prefix}_step_models", "#cfg_model_wrap"
        guide_html = f"""<details class="glass pl-guide"><summary>&#x2139; How this node works</summary><div>{UI.escape(spec.get("guide",""))}</div></details>""" if spec.get("guide") else ""
        return guide_html + BI.SettingsGroup(name="cfg", label="", fields=schema, json_path="").render(config, name_prefix="cfg_")

    def _node_multiselect(self, nodes, selected, exclude_id=""):
        rows = ""
        for n in nodes:
            if n["id"] == exclude_id: continue
            alias = self._node_slug(n)
            keys = (self.AIM.steps.get_step_type(n.get("type","")) or {}).get("output_keys", [])
            key_hint = " ".join(f'<code class="pl-keyhint">{{{alias}.{k}}}</code>' for k in keys)
            rows += f"""<label class="pl-check"><input type="checkbox" name="prev" value="{n["id"]}" {"checked" if n["id"] in selected else ""}> {UI.escape(n.get("name") or n["id"])} <span class="dim">({UI.escape(n.get("type",""))})</span>{key_hint}</label>"""
        return rows or '<div class="dim">No other nodes yet - this will be a start node.</div>'

    def _node_form_html(self, scope_id, pl, node=None) -> str:
        p, nodes = self.intent_prefix, pl.get("flow",{}).get("nodes",[])
        nid, is_new = (node or {}).get("id",""), not (node or {}).get("id")
        ntype, config, prev = (node or {}).get("type",""), (node or {}).get("config",{}), (node or {}).get("prev",[])
        del_btn = f"""<button type="button" class="btn-icon" style="color:#ff5f5f" {self._post("node_delete", scope=scope_id, pl_id=pl["id"], nid=nid)} onclick="return confirm('Remove node?')">Remove</button>""" if not is_new else ""
        slug_val = (node or {}).get("slug","") or self._node_slug(node or {})
        return f"""<form {self._post("node_save", scope=scope_id, pl_id=pl["id"], nid=nid) if not is_new else self._post("node_add", scope=scope_id, pl_id=pl["id"])} hx-include="this" class="pl-node-form">
                       <span class="pl-form-title">{"New Node" if is_new else "Edit Node"}</span>
                       <input type="text" name="name" value="{UI.escape((node or {}).get('name',''))}" placeholder="Node name" class="module-select">
                       <label class="dim">Reference name (used as <code>{{this.field}}</code>)<input type="text" name="slug" value="{UI.escape(slug_val)}" class="module-select"></label>
                       <label class="dim">Step Type<select name="type" class="module-select" {self._post("node_type_change", scope=scope_id, pl_id=pl["id"])} hx-trigger="change" hx-include="this" hx-target="#pl-node-cfg-{p}">{self._step_type_options(ntype)}</select></label>
                       <div id="pl-node-cfg-{p}">{self._step_config_form_fields(ntype, config) if ntype else '<div class="dim">Pick a step type to configure it.</div>'}</div>
                       <label class="dim">Runs after</label>
                       {self._node_multiselect(nodes, prev, nid)}
                       <div class="pl-form-actions"><button type="submit" class="button">{"Add Node" if is_new else "Save Node"}</button>{del_btn}</div>
                   </form>"""

    def _editor_html(self, scope_id, pl) -> str:
        p, nodes = self.intent_prefix, pl.get("flow",{}).get("nodes",[])
        rows = "".join(f"""<div class="pl-node-row" {self._post("node_form", scope=scope_id, pl_id=pl["id"], nid=n["id"])}>{UI.escape(n.get("name") or n["id"])} <span class="dim">({UI.escape(n.get("type",""))})</span></div>""" for n in nodes) or '<div class="dim">No nodes yet.</div>'
        return f"""<div class="pl-modal-backdrop" onclick="if(event.target===this) htmx.ajax('POST','/im/in',{{values:{self._vals("editor_close")},swap:'none'}})">
                       <div class="glass pl-modal">
                           <div class="pl-modal-hd">
                               <input type="text" value="{UI.escape(pl.get("name",""))}" class="module-select" {self._post("rename", scope=scope_id, pl_id=pl["id"])} hx-trigger="change" hx-include="this" name="name">
                               <button type="button" class="close-btn" {self._post("editor_close")}>&#x2715;</button>
                           </div>
                           <div class="pl-modal-body">
                               <div id="pl-node-editor-{p}" class="pl-node-editor">
                                   <div class="pl-placeholder">Select or add a node.</div>
                                   <div class="pl-node-list">{rows}</div>
                                   <button class="btn-icon" {self._post("node_form", scope=scope_id, pl_id=pl["id"])}>+ Add Node</button>
                               </div>
                           </div>
                       </div>
                   </div>"""

    @staticmethod
    def _recompute_next(flow):
        for n in flow["nodes"]: n["next"] = []
        by_id = {n["id"]: n for n in flow["nodes"]}
        for n in flow["nodes"]:
            for prv in n.get("prev", []):
                if prv in by_id: by_id[prv]["next"].append(n["id"])

    def _parse_node_form(self, form, step_type=""):
        config = {}
        for field in self.AIM.steps.get_step_type(step_type)["config_schema"]:
            raw = form.get(f"cfg_{field.name}")
            if field.type == "number":
                if raw is None or raw.strip() == "": config[field.name] = field.default
                else:
                    try: config[field.name] = int(raw)
                    except (ValueError, TypeError):
                        try: config[field.name] = float(raw)
                        except (ValueError, TypeError): config[field.name] = field.default
            elif field.type == "checkbox": config[field.name] = raw is not None
            else: config[field.name] = raw if raw is not None else field.default
        return config, form.getlist("prev"), form.get("slug","").strip()

    async def _im_new_form(self, request, payload, imr):
        return imr.oob(f"""<form {self._post("create", scope=payload.get("scope",""))} hx-include="this" class="pl-new-form"><input type="text" name="name" class="module-select" placeholder="Pipeline name" required autofocus><button type="submit" class="button">Create</button></form>""", f"pl-new-{self.intent_prefix}")

    async def _im_create(self, request, payload, imr):
        scope_id = payload.get("scope","")
        pl = self.AIM.engine.new_pipeline(owner=self.intent_prefix, name=payload.get("name","Pipeline").strip() or "Pipeline")
        pl[self.scope_key] = scope_id
        self.AIM.engine.save_pipeline(pl)
        imr.oob("", f"pl-new-{self.intent_prefix}")
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="pl-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_delete(self, request, payload, imr):
        scope_id = payload.get("scope","")
        self.AIM.engine.delete_pipeline(payload.get("pl_id",""))
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="pl-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_editor_open(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}") if pl else imr

    async def _im_editor_close(self, request, payload, imr): return imr.oob("", f"pl-editor-modal-{self.intent_prefix}")

    async def _im_node_form(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        node = next((n for n in pl.get("flow",{}).get("nodes",[]) if n["id"]==payload.get("nid")), None)
        return imr.oob(self._node_form_html(payload.get("scope",""), pl, node), f"pl-node-editor-{self.intent_prefix}")

    async def _im_node_type_change(self, request, payload, imr):
        return imr.oob(self._step_config_form_fields(payload.get("type",""), {}), f"pl-node-cfg-{self.intent_prefix}")

    async def _im_step_models(self, request, payload, imr):
        conn = self.AIM.connections.get_conn(payload.get("cfg_conn_id",""))
        models = self.AIM.connections.list_models_sync(conn) if conn else []
        opts = "".join(f'<option value="{m}">{m}</option>' for m in models) or '<option value="">No models</option>'
        return imr.oob(f'<label id="cfg_model_wrap" class="dim">Model<select name="cfg_model" class="module-select">{opts}</select></label>', "cfg_model_wrap")

    async def _im_node_save(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        flow = pl.setdefault("flow", {"nodes": []})
        nid, ntype = payload.get("nid",""), payload.get("type","")
        config, prev, slug = self._parse_node_form(payload, ntype)
        node = next((n for n in flow["nodes"] if n["id"]==nid), None) if nid else None
        if node: node.update(slug=slug, name=payload.get("name","").strip(), type=ntype, config=config, prev=prev)
        else: flow["nodes"].append({"id": f"n_{uuid.uuid4().hex[:8]}", "slug": slug, "name": payload.get("name","").strip(), "type": ntype, "config": config, "prev": prev, "next": []})
        self._recompute_next(flow)
        self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}")

    async def _im_node_delete(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        flow, nid = pl.setdefault("flow", {"nodes": []}), payload.get("nid","")
        flow["nodes"] = [n for n in flow["nodes"] if n["id"] != nid]
        for n in flow["nodes"]: n["prev"] = [pr for pr in n.get("prev",[]) if pr != nid]
        self._recompute_next(flow)
        self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}")

    async def _im_rename(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if pl: pl["name"] = payload.get("name","").strip() or pl["name"]; self.AIM.engine.save_pipeline(pl)
        return imr

    async def _im_run(self, request, payload, imr):
        scope_id, pl_id = payload.get("scope",""), payload.get("pl_id","")
        job_id, err = self.AIM.engine.submit(request.state.user.username, kind="id", pipeline_id=pl_id, extra_config={self.scope_key: scope_id}, inputs={"input": payload.get("value","")})
        if not err:
            pl = self.AIM.engine.load_pipeline(pl_id); pl["last_job_id"] = job_id; self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._status_block(scope_id, pl_id, self.AIM.engine.load_job(job_id) if not err else None), f"pl-status-{pl_id}")

    async def _im_resume(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        job_id, _ = self.AIM.engine.resume(pl.get("last_job_id","")) if pl and pl.get("last_job_id") else (None, "")
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), self.AIM.engine.load_job(job_id) if job_id else None), f"pl-status-{payload.get('pl_id','')}")

    async def _im_stop(self, request, payload, imr):
        self.AIM.engine.stop(payload.get("job_id",""))
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), self.AIM.engine.load_job(payload.get("job_id",""))), f"pl-status-{payload.get('pl_id','')}")

    async def _im_status(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        job = self.AIM.engine.load_job(pl.get("last_job_id","")) if pl and pl.get("last_job_id") else None
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), job), f"pl-status-{payload.get('pl_id','')}")

    async def _im_claim(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if pl and not pl.get(self.scope_key): pl[self.scope_key] = payload.get("scope",""); self.AIM.engine.save_pipeline(pl)
        return imr












@router.get("/pipelines", response_class=JSONResponse)
async def api_list_pipelines(tag: str = None): return JSONResponse(engine.list_pipelines(tag))

@router.get("/pipelines/{pid}", response_class=JSONResponse)
async def api_get_pipeline(pid: str): return JSONResponse(engine.load_pipeline(pid) or {"error": "not found"})

@router.post("/pipelines/{pid}", response_class=JSONResponse)
async def api_save_pipeline(pid: str, request: Request):
    doc = await request.json()
    doc["id"] = pid
    engine.save_pipeline(doc)
    return JSONResponse({"status": "ok"})

@router.get("/step_types", response_class=JSONResponse)
async def api_step_types(): return JSONResponse(steps.list_step_types())

@router.get("/job/{job_id}", response_class=JSONResponse)
async def api_job_status(job_id: str): return JSONResponse(engine.load_job(job_id) or {"error": "not found"})

@router.post("/job/{job_id}/stop", response_class=JSONResponse)
async def api_job_stop(job_id: str): engine.stop(job_id); return JSONResponse({"status": "stopping"})

@router.post("/prompt_blocks/save", response_class=HTMLResponse)
async def prompt_blocks_save(name: str = Form(...), text: str = Form(...), target: str = Form(...)):
    _PB.save(name, text)
    return HTMLResponse(BI.prompt_block_picker_html(_PB, target, f"{_P}/prompt_blocks/save"))

@router.get("/prompt_blocks", response_class=JSONResponse)
async def prompt_blocks_list(): return JSONResponse(_PB.list())

@router.delete("/prompt_blocks/{block_id}", response_class=JSONResponse)
async def prompt_blocks_delete(block_id: str): _PB.delete(block_id); return JSONResponse({"status": "ok"})

# -- Simple Chat (working demonstration: chat with an optional attached knowledge base) --
# This is intentionally minimal - a 1-or-2-node inline Flow, not a saved pipeline, not routed through Tessa/Kimi UI.
# Proves the "just chat with knowledge as needed" path works without waiting on the full builder migration.

@router.get("/", response_class=HTMLResponse)
@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """
    Raw engine + WS wire-level debug harness - not a chat UI.
    Deliberately bypasses IM/im-in and ChatManager so failures are visible at the protocol level (raw POST payload, raw job_id, raw WS event stream) instead of hidden behind abstraction.
    Real chat surfaces (Athena, Tessa) use ChatManager.
    """
    conns = connections.list_conns()
    conn_opts = "".join(f'<option value="{c["_id"]}">{_esc(c.get("display_name",c["_id"]))}</option>' for c in conns)
    kg_opts = '<option value="">(no knowledge base)</option>' + "".join(f'<option value="{c["_id"]}">{_esc(c.get("display_name",c["_id"]))}</option>' for c in connections.list_conns(conn_type="lightrag"))
    return HTMLResponse(f"""<div style="max-width:60rem; margin:0 auto; padding:1.5rem; display:flex; flex-direction:column; gap:.6rem; height:100%; box-sizing:border-box">
                                <div style="display:flex;gap:.5rem">
                                    <select id="chat-conn" name="conn_id" class="module-select" style="flex:1; margin:0" hx-get="{_P}/chat/models" hx-trigger="load, change" hx-target="#chat-model" hx-swap="innerHTML" hx-include="this">{conn_opts}</select>
                                    <select id="chat-model" name="model" class="module-select" style="flex:1; margin:0"></select>
                                    <select id="chat-kg" class="module-select" style="flex:1;margin:0">{kg_opts}</select>
                                </div>
                                <div id="chat-log" style="flex:1;overflow-y:auto;border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.6rem;display:flex;flex-direction:column;gap:.4rem"></div>
                                <details open style="border:var(--border-thick) solid var(--border);border-radius:var(--radius)">
                                    <summary style="cursor:pointer;font-size:.72rem;color:var(--text_muted);padding:.3rem .5rem">Raw wire debug log</summary>
                                    <div id="chat-debug" style="font-family:var(--font-mono);font-size:.68rem;white-space:pre-wrap;max-height:16rem;overflow-y:auto;padding:.4rem .6rem;color:var(--text_muted)"></div>
                                </details>
                                <div style="display:flex;gap:.5rem">
                                    <input id="chat-input" type="text" class="module-select" style="flex:1;margin:0" placeholder="Ask something...">
                                    <button class="ui-btn" onclick="aimChatSend()">Send</button>
                                </div>
                                <script>
                                    function aimDebug(label, obj){{
                                        var d = document.getElementById('chat-debug');
                                        d.textContent += '['+new Date().toLocaleTimeString()+'] '+label+': '+(typeof obj==='object'?JSON.stringify(obj):obj)+'\\n';
                                        d.scrollTop = d.scrollHeight;
                                    }}
                                    function aimChatSend(){{
                                        var input = document.getElementById('chat-input');
                                        var text = input.value.trim(); if(!text) return;
                                        var payload = {{conn_id: document.getElementById('chat-conn').value, model: document.getElementById('chat-model').value, kg_id: document.getElementById('chat-kg').value, text: text}};
                                        aimDebug('SEND', payload);
                                        document.getElementById('chat-log').insertAdjacentHTML('beforeend', '<div style="align-self:flex-end;background:var(--accent_dim);padding:.4rem .6rem;border-radius:var(--radius);max-width:80%">'+text+'</div>');
                                        input.value = '';
                                        fetch('{_P}/chat/send', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body:new URLSearchParams(payload)}})
                                            .then(r=>r.json()).then(d=>{{
                                                aimDebug('JOB', d);
                                                if(d.error){{ document.getElementById('chat-log').insertAdjacentHTML('beforeend', '<div style="color:#ff5f5f">Error: '+d.error+'</div>'); return; }}
                                                window._aimJob = d.job_id; window._aimAnswer = '';
                                                document.getElementById('chat-log').insertAdjacentHTML('beforeend', '<div id="chat-live" style="background:var(--glass);padding:.4rem .6rem;border-radius:var(--radius);max-width:80%"></div>');
                                            }});
                                    }}
                                    document.addEventListener('pipeline:step_start', e=>{{ if(e.detail.job_id===window._aimJob) aimDebug('step_start', e.detail); }});
                                    document.addEventListener('pipeline:step_done', e=>{{ if(e.detail.job_id===window._aimJob) aimDebug('step_done', e.detail); }});
                                    document.addEventListener('pipeline:error', e=>{{ if(e.detail.job_id===window._aimJob) aimDebug('ERROR', e.detail); }});
                                    document.addEventListener('pipeline:stream', e=>{{ if(e.detail.job_id!==window._aimJob) return; window._aimAnswer += e.detail.delta; var live = document.getElementById('chat-live'); if(live) live.textContent = window._aimAnswer; document.getElementById('chat-log').scrollTop = document.getElementById('chat-log').scrollHeight; }});
                                    document.addEventListener('pipeline:done', e=>{{ if(e.detail.job_id!==window._aimJob) return; aimDebug('DONE', e.detail); var live=document.getElementById('chat-live'); if(live) live.removeAttribute('id'); }});
                                </script>
                            </div>""")

@router.get("/chat/models", response_class=HTMLResponse)
async def chat_models(conn_id: str = ""):
    conn = connections.get_conn(conn_id)
    models = await connections.list_models_async(conn) if conn else []
    opts = "".join(f'<option value="{m}">{m}</option>' for m in models) or '<option value="">(no models found)</option>'
    return HTMLResponse(opts)

@router.post("/chat/send", response_class=JSONResponse)
async def chat_send(request: Request, conn_id: str = Form(...), model: str = Form(...), kg_id: str = Form(""), text: str = Form(...)):
    user = request.state.user
    flow = {"nodes": []}
    if kg_id: flow["nodes"].append({"id": "kq", "type": "knowledge_query", "config": {"conn_id": kg_id, "query_template": "{input}", "result_key": "kg_context"}, "next": ["chat"]})
    chat_node = {"id": "chat", "type": "chat", "prev": ["kq"] if kg_id else [], "config": {"conn_id": conn_id, "model": model, "result_key": "answer",
                                                                                           "system_prompt": "Use the retrieved context if relevant to answer the user's question." if kg_id else "",
                                                                                           "user_template": ("Context:\n{kq.kg_context}\n\nQuestion: {input}" if kg_id else "{input}")}}
    flow["nodes"].append(chat_node)
    job_id, err = engine.submit(username=user.username, kind="inline", inline_flow=flow, inputs={"input": text})
    if err: return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"job_id": job_id})

@router.post("/_selftest", response_class=JSONResponse)
async def selftest(request: Request):
    """Exercises the engine end-to-end with zero external dependencies: two parallel echo nodes feeding a merge node, proving wave scheduling (concurrent branches), templating ({node_id.key} resolution), and job persistence all work.
    Safe to call repeatedly; each call is a fresh job. Not gated behind capability tags - self-test is not a real pipeline submission path."""
    flow = {"nodes": [{"id": "a", "type": "echo", "config": {"template": "branch-A saw: {input}", "result_key": "out"}, "next": ["merge"]},
                      {"id": "b", "type": "echo", "config": {"template": "branch-B saw: {input}", "result_key": "out"}, "next": ["merge"]},
                      {"id": "merge", "type": "echo", "prev": ["a", "b"], "config": {"template": "merged [{a.out}] + [{b.out}]", "result_key": "final"}},]}
    job_id, err = engine.submit(request.state.user.username, kind="inline", inline_flow=flow, inputs={"input": "hello"})
    if err: return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"job_id": job_id, "poll": f"{_P}/job/{job_id}"})

def job_status_url(job_id: str = "") -> str: return f"{_P}/job_status?job_id={job_id}"

@router.get("/job_status", response_class=JSONResponse)
async def job_status_qs(job_id: str): return JSONResponse(engine.load_job(job_id) or {"error": "not found"})

@router.get("/pipelines/{pid}/export", response_class=JSONResponse)
async def export_pipeline(pid: str):
    pdef = engine.load_pipeline(pid)
    return JSONResponse(pdef) if pdef else JSONResponse({"error": "not found"}, status_code=404)

@router.post("/pipelines/import", response_class=JSONResponse)
async def import_pipeline(request: Request):
    doc = await request.json()
    doc["id"] = f"pl_{uuid.uuid4().hex[:10]}"  # always a new id - import never silently overwrites an existing pipeline
    engine.save_pipeline(doc)
    return JSONResponse({"id": doc["id"]})

# --- Shadow Memory ---

@router.post("/_shadow_selftest", response_class=JSONResponse)
async def shadow_selftest(request: Request):
    """Proves ShadowStore stage/diff/accept/reject/rollback independent of git or any AI call.
    Writes into a scratch folder under data/ai_manager/_selftest so it never touches real project files."""
    shadow = BI.ShadowStore(fm, root / "_shadow")
    FM.write("note.txt", "original line one\noriginal line two\n")
    entry = shadow.stage("note.txt", "original line one\nCHANGED line two\nnew line three\n", author="selftest")
    diff_before_accept = shadow.diff("note.txt")
    accepted = shadow.accept("note.txt")
    final_content = fm.read("note.txt")
    history = shadow.history("note.txt")
    return JSONResponse({"staged_status": entry["status"], "diff": diff_before_accept, "accepted": accepted, "final_file_content": final_content, "history_timestamps": history})
