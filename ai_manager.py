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
from tools.ai_manager import resources
from tools.ai_manager.flow import Flow

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
IM = globals().get("IM", None)
_NAMED_ROOTS = {"common": "./data/_common"}

def register_root(name: str, path: str): _NAMED_ROOTS[name] = path
def resolve_root(name: str) -> str: return _NAMED_ROOTS.get(name, name)
def list_roots() -> list: return list(_NAMED_ROOTS.items())

def init_module(env: dict):
    global ENV, UI, BI, FM, _PB, IM
    ENV.update(env)
    UI = ENV["templates"].env.globals.get("UI")
    BI = ENV["tools"]["built_ins"]
    FM = BI.FileManager(TOOL_ROOT)
    _PB = BI.PromptBlockLibrary(PROMPT_BLOCKS_PATH)
    IM = ENV["InterfaceManager"](nesting_level=1, db_path="ai_manager/im_registry.db")
    engine.init(env)
    steps.init(env)
    steps.register_builtins()
    IM.scripts.update({"cnode_save": [_im_cnode_save], "cnode_delete": [_im_cnode_delete], "cnode_edit_form": [_im_cnode_edit_form], "resources_open": [_im_resources_open]})
    print(f"[ai_manager] ready | node types: {[t['type'] for t in steps.list_node_types()]}")

def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")
def prompt_block_picker_fragment(textarea_id: str) -> str: return BI.prompt_block_picker_html(_PB, textarea_id, f"{_P}/prompt_blocks/save")

class PipelineBuilderUI:
    """Universal-node pipeline authoring surface. No prev/next wiring anywhere in this UI - node order is entirely a function of key presence, computed fresh for the graphical view via Flow.resolve_levels()."""

    def __init__(self, IM, AIM, intent_prefix="plb", nesting_level=2, scope_key="project_id"):
        self.IM, self.AIM, self.intent_prefix, self.nesting_level, self.scope_key = IM, AIM, intent_prefix, nesting_level, scope_key
        p = self.intent_prefix
        IM.scripts.update({f"{p}_new_form": [self._im_new_form], f"{p}_create": [self._im_create], f"{p}_delete": [self._im_delete], f"{p}_import": [self._im_import], f"{p}_editor_open": [self._im_editor_open], f"{p}_view_toggle": [self._im_view_toggle], f"{p}_node_form": [self._im_node_form], f"{p}_node_type_change": [self._im_node_type_change], f"{p}_node_add": [self._im_node_save], f"{p}_node_save": [self._im_node_save], f"{p}_node_delete": [self._im_node_delete], f"{p}_rename": [self._im_rename], f"{p}_run": [self._im_run], f"{p}_stop": [self._im_stop], f"{p}_resume": [self._im_resume], f"{p}_status": [self._im_status], f"{p}_pool_form": [self._im_pool_form], f"{p}_pool_save": [self._im_pool_save], f"{p}_node_conn_change": [self._im_node_conn_change],})

    def _vals(self, action, **extra): return json.dumps({"type": f"{self.intent_prefix}_{action}", "branch": self.intent_prefix, "lvl": self.nesting_level, **extra})
    def _post(self, action, **extra): return f"""hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{self._vals(action, **extra)}'"""
    def _pipelines(self, scope_id): return [p for p in self.AIM.engine.list_pipelines() if p.get(self.scope_key) == scope_id]

    def _card_html(self, scope_id, pl) -> str:
        pl_id = pl["id"]
        last_job = self.AIM.engine.load_job(pl.get("last_job_id","")) if pl.get("last_job_id") else None
        nodes = (last_job["flow"]["nodes"] if last_job else pl.get("flow",{}).get("nodes",[]))
        pool = pl.get("pool", self.AIM.engine.DEFAULT_POOL)
        pool_badge = f'<span class="status-badge" title="whitelist:{",".join(pool.get("whitelist_tags",[])) or "any"} blacklist:{",".join(pool.get("blacklist_tags",[]))}">{pool.get("priority","balanced")}</span>'
        rows = "".join(self._node_status_row(n) for n in nodes)
        return f"""<div class="glass list-card">
                       <div class="list-card-hd">
                           <span class="list-card-title" {self._post("editor_open", scope=scope_id, pl_id=pl_id)}>{UI.escape(pl.get("name",""))}</span>
                           <span class="dim tiny" style="font-family:var(--font-mono)">{pl_id}</span>
                           {pool_badge}
                           <a class="cm-qbtn" href="{_P}/pipelines/{pl_id}/export" download="{pl_id}.json" title="Export pipeline as JSON">&#x2B07;</a>
                           <button class="btn-icon" style="color:#ff5f5f" {self._post("delete", scope=scope_id, pl_id=pl_id)} onclick="return confirm('Delete pipeline?')">&#x2715;</button>
                       </div>
                       <input type="text" id="pl-input-{pl_id}" name="value" placeholder="Input for this run" class="module-select">
                       <div id="pl-status-{pl_id}">{self._status_block(scope_id, pl_id, last_job)}</div>
                       <details class="status-details"><summary>Node status (<span id="pl-nodecount-{pl_id}">{len(nodes)}</span>)</summary><table class="status-table" id="pl-nodetable-{pl_id}">{rows}</table></details>
                   </div>"""

    def _node_status_row(self, n) -> str:
        col = {"done":"#00ffa2","error":"#ff5f5f","running":"#ffcc00","unreached":"var(--text_muted)"}.get(n.get("status","idle"), "var(--text)")
        preview = n.get("message") or " | ".join(str(v) for v in (n.get("preview") or {}).values())
        return f"""<tr><td class="qn">{UI.escape(n.get("name") or n["id"])}</td><td class="dim">{UI.escape(n.get("type",""))}</td><td style="color:{col}">{UI.escape(n.get("status","idle"))}</td><td class="dim status-preview">{UI.escape(preview)}</td></tr>"""

    def panel_html(self, scope_id: str, include_modal_slot: bool = True) -> str:
        p = self.intent_prefix
        cards = "".join(self._card_html(scope_id, pl) for pl in self._pipelines(scope_id)) or '<div class="list-empty">No pipelines. Click + to create one.</div>'
        modal_slot = f'<div id="pl-editor-modal-{p}"></div>' if include_modal_slot else ""
        return f"""<div id="pl-panel-{p}" class="list-panel">
                       <div class="list-panel-hd"><span class="list-panel-title">Pipelines</span><button class="btn-icon" {self._post("new_form", scope=scope_id)}>+</button></div>
                       <div id="pl-new-{p}"></div>
                       <details class="list-new-form" style="margin:.3rem 0"><summary style="cursor:pointer;font-size:.72rem;color:var(--text_muted)">Import (paste JSON)</summary>
                           <form {self._post("import", scope=scope_id)} hx-include="this" style="display:flex;flex-direction:column;gap:.3rem;padding:.3rem 0">
                               <textarea name="json" class="cm-input" rows="4" placeholder="Paste exported pipeline JSON here"></textarea>
                               <button type="submit" class="button">Import</button>
                               <span id="pl-import-msg-{p}"></span>
                           </form>
                       </details>
                       <div id="pl-list-{p}" class="list-body">{cards}</div>
                       {modal_slot}
                   </div>"""

    def modal_slot_html(self, scope_id: str = "") -> str: return f'<div id="pl-editor-modal-{self.intent_prefix}"></div>' # Render at a non-transformed DOM level (main content area, not inside a sliding toolbar) - position:fixed modal backdrops are trapped inside any ancestor with an active CSS transform.

    def _list_view_html(self, scope_id, pl) -> str:
        p = self.intent_prefix
        rows = "".join(f"""<div class="node-row" {self._post("node_form", scope=scope_id, pl_id=pl["id"], nid=n["id"])}>{UI.escape(n.get("name") or n["id"])} <span class="dim">({UI.escape(n.get("type",""))})</span></div>""" for n in pl.get("flow",{}).get("nodes",[])) or '<div class="dim">No nodes yet.</div>'
        return f"""<div id="pl-node-editor-{p}" class="node-editor">
                       <div class="list-placeholder">Select or add a node.</div>
                       <div class="node-list">{rows}</div>
                       <button class="btn-icon" {self._post("node_form", scope=scope_id, pl_id=pl["id"])}>+ Add Node</button>
                   </div>"""

    def _graph_view_html(self, pl) -> str:
        f = Flow(pl.get("flow", {}))
        type_specs = {t["type"]: t for t in self.AIM.steps.list_node_types()}
        levels = f.resolve_levels(type_specs)
        by_level: dict = {}
        for n in f.nodes.values(): by_level.setdefault(levels.get(n.id, 0), []).append(n)
        rows = "".join(f"""<div class="node-graph-row"><span class="node-graph-row-lbl">{lvl}</span>{"".join(f'<div class="node-block" {self._post("node_form", scope=pl.get(self.scope_key,""), pl_id=pl["id"], nid=n.id)}><b>{UI.escape(n.name or n.id)}</b><span class="dim">{UI.escape(n.type)}</span></div>' for n in nodes)}</div>""" for lvl, nodes in sorted(by_level.items()))
        return f'<div class="node-graph">{rows or "<div class=dim>No nodes yet.</div>"}</div>'

    def _editor_html(self, scope_id, pl, view: str = "list") -> str:
        p = self.intent_prefix
        body = self._graph_view_html(pl) if view == "graph" else self._list_view_html(scope_id, pl)
        toggle = "".join(f"""<button class="cm-qbtn {"active" if view==v else ""}" {self._post("view_toggle", scope=scope_id, pl_id=pl["id"], view=v)}>{lbl}</button>""" for v,lbl in (("list","List"),("graph","Graph")))
        header = f"""<div style="display:flex;gap:.4rem;align-items:center;margin-bottom:.5rem">
                         <input type="text" value="{UI.escape(pl.get("name",""))}" class="module-select" {self._post("rename", scope=scope_id, pl_id=pl["id"])} hx-trigger="change" hx-include="this" name="name" style="flex:1">
                         {toggle}
                         <button type="button" class="btn-icon" {self._post("pool_form", scope=scope_id, pl_id=pl["id"])} title="Resource pool">Pool</button>
                     </div>
                     <div id="pl-pool-{p}"></div>"""
        return UI.modal(f"{p}-editor", "Pipeline Editor", header + body, width="90%", max_width="70rem")

    def _pool_form_html(self, scope_id, pl) -> str:
        pool = pl.get("pool", self.AIM.engine.DEFAULT_POOL)
        all_tags = sorted({t for c in self.AIM.resources.list_cnodes() for t in c.get("tags",[])})
        tag_opts = lambda selected: "".join(f'<option value="{t}" {"selected" if t in selected else ""}>{t}</option>' for t in all_tags)
        return f"""<form {self._post("pool_save", scope=scope_id, pl_id=pl["id"])} hx-include="this" class="glass" style="padding:.6rem;margin:.3rem 0;display:flex;flex-direction:column;gap:.4rem">
                       <label class="dim">Priority<select name="priority" class="module-select">{"".join(f'<option value="{v}" {"selected" if v==pool.get("priority","balanced") else ""}>{l}</option>' for v,l in (("speed","Speed"),("balanced","Balanced"),("quality","Quality")))}</select></label>
                       <label class="dim">Whitelist Tags (empty = any)<select name="whitelist_tags" multiple size="4" class="module-select">{tag_opts(pool.get("whitelist_tags",[]))}</select></label>
                       <label class="dim">Blacklist Tags<select name="blacklist_tags" multiple size="4" class="module-select">{tag_opts(pool.get("blacklist_tags",[]))}</select></label>
                       <button type="submit" class="button">Save Pool</button>
                   </form>"""

    async def _im_view_toggle(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        return imr.oob(self._editor_html(payload.get("scope",""), pl, view=payload.get("view","list")), f"pl-editor-modal-{self.intent_prefix}") if pl else imr

    async def _im_import(self, request, payload, imr):
        scope_id = payload.get("scope","")
        try: doc = json.loads(payload.get("json","{}"))
        except Exception as e:
            imr.raw(f'<span id="pl-import-msg-{self.intent_prefix}" style="color:#ff5f5f;font-size:.7rem" hx-swap-oob="outerHTML">Invalid JSON: {UI.escape(str(e))}</span>')
            return imr
        doc["id"] = f"pl_{uuid.uuid4().hex[:10]}"
        doc[self.scope_key] = scope_id
        doc.setdefault("pool", self.AIM.engine.DEFAULT_POOL)
        doc.setdefault("created", "")
        self.AIM.engine.save_pipeline(doc)
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="list-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_pool_form(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        return imr.oob(self._pool_form_html(payload.get("scope",""), pl), f"pl-pool-{self.intent_prefix}") if pl else imr

    async def _im_pool_save(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        wl, bl = payload.get("whitelist_tags", []), payload.get("blacklist_tags", [])
        pl["pool"] = {"priority": payload.get("priority","balanced"), "whitelist_tags": wl if isinstance(wl, list) else ([wl] if wl else []), "blacklist_tags": bl if isinstance(bl, list) else ([bl] if bl else []), "whitelist_cnodes": pl.get("pool",{}).get("whitelist_cnodes",[]), "blacklist_cnodes": pl.get("pool",{}).get("blacklist_cnodes",[])}
        self.AIM.engine.save_pipeline(pl)
        scope_id = payload.get("scope","")
        imr.oob("", f"pl-pool-{self.intent_prefix}")
        imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="list-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")
        return imr

    def _pool_preview_html(self, pl, tags_str) -> str:
        tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]
        candidates = self.AIM.resources.resolve_candidates(pl.get("pool", self.AIM.engine.DEFAULT_POOL), tags)
        if not candidates: return '<div class="glass" style="padding:.4rem .6rem;font-size:.7rem;color:#ff9944">&#x26A0; No CNodes match this pool + tags - the node will fail at run time unless a connection is pinned above.</div>'
        rows = "".join(f'<div style="font-size:.7rem;padding:.15rem 0;border-bottom:1px solid var(--border)"><b>{UI.escape(c["label"])}</b> <span class="dim">[{UI.escape(", ".join(c.get("tags",[])))}]</span> - {UI.escape(", ".join((self.AIM.connections.load_conn_raw(cid) or {}).get("display_name",cid) for cid in c.get("conn_ids",[])) or "no connections")}</div>' for c in candidates)
        return f'<div class="glass" style="padding:.4rem .6rem"><div style="font-size:.65rem;color:var(--text_muted);text-transform:uppercase;margin-bottom:.2rem">Matching CNodes ({len(candidates)})</div>{rows}</div>'

    def _node_config_fields(self, node_type, config, pl=None):
        spec = self.AIM.steps.get_node_type(node_type)
        if not spec: return '<div class="dim">Pick a node type to configure it.</div>'
        schema = [copy.copy(f) if f.name in ("conn_id","cnode_tags") else f for f in spec["config_schema"]]
        for f in schema:
            if f.name == "conn_id": f.hx_intent, f.hx_target = f"{self.intent_prefix}_node_conn_change", "#cfg_model_wrap"
            if f.name == "cnode_tags": f.hx_intent, f.hx_target = f"{self.intent_prefix}_node_pool_preview", f"#pl-pool-preview-{self.intent_prefix}"
        guide_html = f"""<details class="glass status-list"><summary>&#x2139; How this node works</summary><div>{UI.escape(spec.get("guide",""))}</div></details>""" if spec.get("guide") else ""
        pool_preview = f'<div id="pl-pool-preview-{self.intent_prefix}" style="margin-bottom:.5rem">{self._pool_preview_html(pl, config.get("cnode_tags",""))}</div>' if (pl is not None and any(f.name == "cnode_tags" for f in schema)) else ""
        return guide_html + pool_preview + BI.SettingsGroup(name="cfg", label="", fields=schema, json_path="").render(config, name_prefix="cfg_")

    async def _im_node_pool_preview(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        return imr.oob(self._pool_preview_html(pl, payload.get("cfg_cnode_tags","")), f"pl-pool-preview-{self.intent_prefix}", swap="outerHTML")

    def _status_block(self, scope_id, pl_id, job) -> str:
        live = bool(job and job.get("status") in ("running","queued"))
        if live:
            return f"""<div class="status-row" hx-trigger="load delay:1.5s" {self._post("status", scope=scope_id, pl_id=pl_id)} hx-target="#pl-status-{pl_id}" hx-swap="innerHTML">
                           <button class="cm-qbtn" style="color:#ff4444" {self._post("stop", scope=scope_id, pl_id=pl_id, job_id=job["id"])}>&#x25FC; Stop</button>
                           <span class="status-label" style="color:#00ffa2">{job["status"]}</span></div>"""
        label = job["status"] if job else "idle"
        resume = f"""<button class="cm-qbtn" {self._post("resume", scope=scope_id, pl_id=pl_id)}>&#x21BB; Resume</button>""" if job and job.get("status") == "interrupted" else ""
        return f"""<div class="status-row">
                       <button class="cm-qbtn" {self._post("run", scope=scope_id, pl_id=pl_id)} hx-include="#pl-input-{pl_id}">&#x25B6; Run</button>
                       {resume}<span class="status-label">{label}</span>
                    </div>"""

    def _node_type_options(self, selected=""):
        blank = '<option value="" selected disabled>-- select node type --</option>' if not selected else ""
        return blank + "".join(f'<option value="{t["type"]}" {"selected" if t["type"]==selected else ""}>{UI.escape(t.get("label",t["type"]))}</option>' for t in self.AIM.steps.list_node_types())

    def _node_form_html(self, scope_id, pl, node=None) -> str:
        p = self.intent_prefix
        nid, is_new = (node or {}).get("id",""), not (node or {}).get("id")
        ntype, config = (node or {}).get("type",""), (node or {}).get("config",{})
        key_map = json.dumps((node or {}).get("key_map",{}), indent=2)
        extra_in, extra_out = ", ".join((node or {}).get("extra_in_keys",[])), ", ".join((node or {}).get("extra_out_keys",[]))
        del_btn = f"""<button type="button" class="btn-icon" style="color:#ff5f5f" {self._post("node_delete", scope=scope_id, pl_id=pl["id"], nid=nid)} onclick="return confirm('Remove node?')">Remove</button>""" if not is_new else ""
        return f"""<form {self._post("node_save", scope=scope_id, pl_id=pl["id"], nid=nid) if not is_new else self._post("node_add", scope=scope_id, pl_id=pl["id"])} hx-include="this" class="node-form">
                       <input type="hidden" name="pl_id" value="{pl["id"]}">
                       <span class="form-title">{"New Node" if is_new else "Edit Node"}</span>
                       <input type="text" name="name" value="{UI.escape((node or {}).get('name',''))}" placeholder="Node name" class="module-select">
                       <label class="dim">Node Type<select name="type" class="module-select" {self._post("node_type_change", scope=scope_id, pl_id=pl["id"])} hx-trigger="change" hx-include="this" hx-target="#pl-node-cfg-{p}">{self._node_type_options(ntype)}</select></label>
                       <div id="pl-node-cfg-{p}">{self._node_config_fields(ntype, config) if ntype else '<div class="dim">Pick a node type to configure it.</div>'}</div>
                       <label class="dim">Extra In Keys (comma-sep, actual pipeline keys)<input type="text" name="extra_in_keys" value="{UI.escape(extra_in)}" class="module-select"></label>
                       <label class="dim">Extra Out Keys (comma-sep, actual pipeline keys)<input type="text" name="extra_out_keys" value="{UI.escape(extra_out)}" class="module-select"></label>
                       <label class="dim">Key Map (JSON: logical name -> actual key, advanced)<textarea name="key_map" class="cm-input" rows="3">{UI.escape(key_map)}</textarea></label>
                       <div class="form-actions"><button type="submit" class="button">{"Add Node" if is_new else "Save Node"}</button>{del_btn}</div>
                   </form>"""

    async def _im_new_form(self, request, payload, imr):
        return imr.oob(f"""<form {self._post("create", scope=payload.get("scope",""))} hx-include="this" class="pl-new-form"><input type="text" name="name" class="module-select" placeholder="Pipeline name" required autofocus><button type="submit" class="button">Create</button></form>""", f"pl-new-{self.intent_prefix}")

    async def _im_create(self, request, payload, imr):
        scope_id = payload.get("scope","")
        pl = self.AIM.engine.new_pipeline(owner=self.intent_prefix, name=payload.get("name","Pipeline").strip() or "Pipeline")
        pl[self.scope_key] = scope_id
        self.AIM.engine.save_pipeline(pl)
        imr.oob("", f"pl-new-{self.intent_prefix}")
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="list-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_delete(self, request, payload, imr):
        scope_id = payload.get("scope","")
        self.AIM.engine.delete_pipeline(payload.get("pl_id",""))
        return imr.oob("".join(self._card_html(scope_id, p_) for p_ in self._pipelines(scope_id)) or '<div class="list-empty">No pipelines.</div>', f"pl-list-{self.intent_prefix}")

    async def _im_editor_open(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}") if pl else imr

    async def _im_node_form(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        node = next((n for n in pl.get("flow",{}).get("nodes",[]) if n["id"]==payload.get("nid")), None)
        return imr.oob(self._node_form_html(payload.get("scope",""), pl, node), f"pl-node-editor-{self.intent_prefix}")

    async def _im_node_type_change(self, request, payload, imr): return imr.oob(self._node_config_fields(payload.get("type",""), {}, self.AIM.engine.load_pipeline(payload.get("pl_id",""))), f"pl-node-cfg-{self.intent_prefix}")

    async def _im_node_save(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        flow = pl.setdefault("flow", {"nodes": []})
        nid, ntype = payload.get("nid",""), payload.get("type","")
        spec = self.AIM.steps.get_node_type(ntype)
        config = {}
        if spec:
            for field in spec["config_schema"]:
                raw = payload.get(f"cfg_{field.name}")
                if field.type == "number": config[field.name] = field.default if raw in (None,"") else (int(raw) if str(raw).lstrip("-").isdigit() else float(raw))
                elif field.type == "checkbox": config[field.name] = raw is not None
                elif field.type == "json":
                    try: config[field.name] = json.loads(raw) if raw and raw.strip() else field.default
                    except Exception: config[field.name] = field.default
                else: config[field.name] = raw if raw is not None else field.default
        try: key_map = json.loads(payload.get("key_map","{}") or "{}")
        except Exception: key_map = {}
        extra_in = [k.strip() for k in payload.get("extra_in_keys","").split(",") if k.strip()]
        extra_out = [k.strip() for k in payload.get("extra_out_keys","").split(",") if k.strip()]
        node = next((n for n in flow["nodes"] if n["id"]==nid), None) if nid else None
        if node: node.update(name=payload.get("name","").strip(), type=ntype, config=config, key_map=key_map, extra_in_keys=extra_in, extra_out_keys=extra_out)
        else: flow["nodes"].append({"id": f"n_{uuid.uuid4().hex[:8]}", "name": payload.get("name","").strip(), "type": ntype, "config": config, "key_map": key_map, "extra_in_keys": extra_in, "extra_out_keys": extra_out, "status": "idle"})
        self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}")

    async def _im_node_delete(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if not pl: return imr
        flow = pl.setdefault("flow", {"nodes": []})
        flow["nodes"] = [n for n in flow["nodes"] if n["id"] != payload.get("nid","")]
        self.AIM.engine.save_pipeline(pl)
        return imr.oob(self._editor_html(payload.get("scope",""), pl), f"pl-editor-modal-{self.intent_prefix}")

    async def _im_rename(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        if pl: pl["name"] = payload.get("name","").strip() or pl["name"]; self.AIM.engine.save_pipeline(pl)
        return imr

    async def _im_run(self, request, payload, imr):
        scope_id, pl_id = payload.get("scope",""), payload.get("pl_id","")
        job_id, err = self.AIM.engine.submit(request.state.user.username, kind="id", pipeline_id=pl_id, inputs={"input": payload.get("value","")})
        job = self.AIM.engine.load_job(job_id) if not err else None
        if not err:
            pl = self.AIM.engine.load_pipeline(pl_id); pl["last_job_id"] = job_id; self.AIM.engine.save_pipeline(pl)
        imr.oob(self._status_block(scope_id, pl_id, job), f"pl-status-{pl_id}")
        imr.oob("".join(self._node_status_row(n) for n in (job["flow"]["nodes"] if job else [])), f"pl-nodetable-{pl_id}")
        return imr

    async def _im_resume(self, request, payload, imr):
        pl = self.AIM.engine.load_pipeline(payload.get("pl_id",""))
        job_id, _ = self.AIM.engine.resume(pl.get("last_job_id","")) if pl and pl.get("last_job_id") else (None, "")
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), self.AIM.engine.load_job(job_id) if job_id else None), f"pl-status-{payload.get('pl_id','')}")

    async def _im_stop(self, request, payload, imr):
        self.AIM.engine.stop(payload.get("job_id",""))
        return imr.oob(self._status_block(payload.get("scope",""), payload.get("pl_id",""), self.AIM.engine.load_job(payload.get("job_id",""))), f"pl-status-{payload.get('pl_id','')}")

    async def _im_status(self, request, payload, imr):
        pl_id = payload.get("pl_id","")
        pl = self.AIM.engine.load_pipeline(pl_id)
        job = self.AIM.engine.load_job(pl.get("last_job_id","")) if pl and pl.get("last_job_id") else None
        imr.oob(self._status_block(payload.get("scope",""), pl_id, job), f"pl-status-{pl_id}")
        nodes = job["flow"]["nodes"] if job else (pl.get("flow",{}).get("nodes",[]) if pl else [])
        imr.oob("".join(self._node_status_row(n) for n in nodes), f"pl-nodetable-{pl_id}")
        return imr

    def _node_config_fields(self, node_type, config):
        spec = self.AIM.steps.get_node_type(node_type)
        if not spec: return '<div class="dim">Pick a node type to configure it.</div>'
        schema = [copy.copy(f) if f.name == "conn_id" else f for f in spec["config_schema"]]
        for f in schema:
            if f.name == "conn_id": f.hx_intent, f.hx_target = f"{self.intent_prefix}_node_conn_change", "#cfg_model_wrap"
        guide_html = f"""<details class="glass status-list"><summary>&#x2139; How this node works</summary><div>{UI.escape(spec.get("guide",""))}</div></details>""" if spec.get("guide") else ""
        return guide_html + BI.SettingsGroup(name="cfg", label="", fields=schema, json_path="").render(config, name_prefix="cfg_")

    async def _im_node_conn_change(self, request, payload, imr):
        conn = self.AIM.connections.get_conn(payload.get("cfg_conn_id",""))
        models = self.AIM.connections.list_models_sync(conn) if conn else []
        opts = "".join(f'<option value="{m}">{m}{" - looks like embedding" if self.AIM.steps.looks_like_embedding(m) else ""}</option>' for m in models)
        return imr.oob(f'<label id="cfg_model_wrap" class="dim">Model<select name="cfg_model" class="module-select"><option value="">(auto)</option>{opts}</select></label>', "cfg_model_wrap")

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
                                <div style="display:flex;justify-content:flex-end">
                                    <button class="ui-btn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"resources_open","lvl":1})}'>Resource Pool (CNodes)</button>
                                </div>
                                <div id="resources-modal-slot"></div>
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
    if kg_id: flow["nodes"].append({"id":"kq","type":"knowledge","config":{"mode":"query","conn_id":kg_id,"query_template":"{input}"},"key_map":{"response":"kg_context"},"status":"idle"})
    flow["nodes"].append({"id":"chat","type":"generate","config":{"conn_id":conn_id,"model":model,"system_prompt":"Use the retrieved context if relevant to answer the user's question." if kg_id else "","user_template":"Context:\n{kg_context}\n\nQuestion: {input}" if kg_id else "{input}"},"extra_in_keys":["kg_context"] if kg_id else [],"key_map":{"text":"answer"},"status":"idle"})
    job_id, err = engine.submit(username=user.username, kind="inline", inline_flow=flow, inputs={"input": text})
    if err: return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"job_id": job_id})

@router.post("/_selftest", response_class=JSONResponse)
async def selftest(request: Request):
    """Exercises the engine end-to-end with zero external dependencies: two parallel echo nodes feeding a merge node, proving wave scheduling (concurrent branches), templating ({node_id.key} resolution), and job persistence all work.
    Safe to call repeatedly; each call is a fresh job. Not gated behind capability tags - self-test is not a real pipeline submission path."""
    flow = {"nodes": [{"id":"a","type":"transform","config":{"mode":"template","template":"branch-A saw: {input}"},"key_map":{"value":"out_a"},"status":"idle"},
                      {"id":"b","type":"transform","config":{"mode":"template","template":"branch-B saw: {input}"},"key_map":{"value":"out_b"},"status":"idle"},
                      {"id":"merge","type":"transform","config":{"mode":"template","template":"merged [{out_a}] + [{out_b}]"},"extra_in_keys":["out_a","out_b"],"key_map":{"value":"final"},"status":"idle"}]}
    job_id, err = engine.submit(request.state.user.username, kind="inline", inline_flow=flow, inputs={"input": "hello"})
    if err: return JSONResponse({"error": err}, status_code=400)
    return JSONResponse({"job_id": job_id, "poll": f"{_P}/job/{job_id}"})

def job_status_url(job_id: str = "") -> str: return f"{_P}/job_status?job_id={job_id}"

@router.get("/job_status", response_class=JSONResponse)
async def job_status_qs(job_id: str): return JSONResponse(engine.load_job(job_id) or {"error": "not found"})

@router.get("/pipelines/{pid}/export", response_class=JSONResponse)
async def export_pipeline(pid: str):
    pdef = engine.load_pipeline(pid)
    if not pdef: return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(pdef, headers={"Content-Disposition": f'attachment; filename="{pid}.json"'})

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

# --- Compute Nodes ---

def _cnode_form_html(cnode: dict = None, cid: str = "") -> str:
    c = cnode or resources.DEFAULT_CNODE
    conn_opts = "".join(f'<option value="{cid2}" {"selected" if cid2 in c.get("conn_ids",[]) else ""}>{cid2} ({conn.get("connection_type","?")})</option>' for f in sorted(connections.CONN_DIR.glob("*.json")) for cid2 in [f.stem] for conn in [connections.load_conn_raw(cid2)] if conn)
    vals = json.dumps({"type":"cnode_save","lvl":1,"cid":cid})
    return f"""<form hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{vals}' hx-include="this" style="display:flex;flex-direction:column;gap:.4rem;padding:.6rem" class="glass">
                   <label class="dim">Label<input type="text" name="label" value="{UI.escape(c.get('label',''))}" class="module-select" required></label>
                   <label class="dim">Tags (comma-sep)<input type="text" name="tags" value="{','.join(c.get('tags',[]))}" class="module-select"></label>
                   <label class="dim">Connections on this CNode<select name="conn_ids" multiple size="5" class="module-select">{conn_opts}</select></label>
                   <label class="dim">Memory (GB)<input type="number" name="mem_gb" value="{c.get('mem_gb',16)}" step="any" class="module-select"></label>
                   <label class="dim">Overhead (GB)<input type="number" name="overhead_gb" value="{c.get('overhead_gb',2)}" step="any" class="module-select"></label>
                   <label class="dim">Compute Score (relative, higher = faster)<input type="number" name="compute_score" value="{c.get('compute_score',1.0)}" step="any" class="module-select"></label>
                   <label class="dim">Quality Score (relative, higher = better)<input type="number" name="quality_score" value="{c.get('quality_score',1.0)}" step="any" class="module-select"></label>
                   <label class="dim">Notes<textarea name="notes" class="cm-input" rows="2">{UI.escape(c.get('notes',''))}</textarea></label>
                   <button type="submit" class="button">Save CNode</button>
               </form>"""

def _cnode_list_html() -> str:
    rows = ""
    for c in resources.list_cnodes():
        conn_names = ", ".join((connections.load_conn_raw(cid) or {}).get("display_name", cid) for cid in c.get("conn_ids", [])) or "no connections"
        rows += f"""<div class="glass" style="padding:.5rem .7rem;margin-bottom:.3rem;display:flex;align-items:center;gap:.5rem">
                        <span style="flex:1;font-weight:600">{UI.escape(c['label'])}</span>
                        <span class="dim tiny">{UI.escape(', '.join(c.get('tags',[])))}</span>
                        <span class="dim tiny">{UI.escape(conn_names)}</span>
                        <button class="cm-qbtn" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"cnode_edit_form","lvl":1,"cid":c["id"]})}'>Edit</button>
                        <button class="cm-qbtn" style="color:#ff5f5f" hx-post="/im/in" hx-target="body" hx-swap="none" hx-vals='{json.dumps({"type":"cnode_delete","lvl":1,"cid":c["id"]})}' hx-confirm="Delete?">&#x2715;</button>
                    </div>"""
    return f'<div id="cnode-list">{rows or "<div class=dim style=padding:.5rem>No CNodes yet.</div>"}</div>'

async def _im_cnode_save(request, payload, imr):
    cid = payload.get("cid","") or f"cnode_{uuid.uuid4().hex[:8]}"
    conn_ids = payload.get("conn_ids", [])
    if isinstance(conn_ids, str): conn_ids = [conn_ids] if conn_ids else []
    resources.save_cnode(cid, {"label": payload.get("label","").strip(), "tags": [t.strip() for t in payload.get("tags","").split(",") if t.strip()], "conn_ids": conn_ids, "mem_gb": float(payload.get("mem_gb") or 16), "overhead_gb": float(payload.get("overhead_gb") or 2), "compute_score": float(payload.get("compute_score") or 1.0), "quality_score": float(payload.get("quality_score") or 1.0), "notes": payload.get("notes","")})
    imr.oob(_cnode_list_html(), "cnode-list", swap="outerHTML")
    imr.oob(_cnode_form_html(), "cnode-form", swap="outerHTML")
    return imr

async def _im_cnode_delete(request, payload, imr):
    resources.delete_cnode(payload.get("cid",""))
    imr.oob(_cnode_list_html(), "cnode-list", swap="outerHTML")
    return imr

async def _im_cnode_edit_form(request, payload, imr):
    imr.oob(_cnode_form_html(resources.get_cnode(payload.get("cid","")), payload.get("cid","")), "cnode-form", swap="outerHTML")
    return imr

def _resources_modal_html() -> str:
    body = f"""<p style="font-size:.8rem;color:var(--text_muted)">Each CNode is one machine hosting one or more connections. A pipeline's pool (whitelist/blacklist tags) plus a node's own cnode_tags narrow this list at run time.</p>
               {_cnode_list_html()}
               <div id="cnode-form" style="margin-top:.8rem">{_cnode_form_html()}</div>"""
    return UI.modal("ai-resources", "Resource Pool - CNodes", body, width="90%", max_width="42rem")

async def _im_resources_open(request, payload, imr): return imr.oob(_resources_modal_html(), "resources-modal-slot")