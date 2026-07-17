#tools/ai_manager
"""
ai_manager — sole centralized tool for AI operations. Owns connections, the step registry, and pipeline execution. 
modules/ai_tools/* (Athena, Kimi, Tessa, Image) are UI surfaces that call into this tool directly (ENV["tools"]["ai_manager"]);
they never hold their own connections or execution loops - avoids the race conditions of tools calling other tools.
"""
import sys, os
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from tools.ai_manager import engine
from tools.ai_manager import steps
# from tools.ai_manager.connections import list_conns, get_conn, list_models_async, conn_opts_html, _base,
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

# -- Pipeline CRUD (consumed by Tessa's builder; kept here since ai_manager owns the data) --

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
    conns = list_conns()
    conn_opts = "".join(f'<option value="{c["_id"]}">{_esc(c.get("display_name",c["_id"]))}</option>' for c in conns)
    kg_opts = '<option value="">(no knowledge base)</option>' + "".join(f'<option value="{c["_id"]}">{_esc(c.get("display_name",c["_id"]))}</option>' for c in list_conns(conn_type="lightrag"))
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
    conn = get_conn(conn_id)
    models = await list_models_async(conn) if conn else []
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

# ai_manager.py — add a plain query-param job-status route (the existing one is path-param only)
@router.get("/job_status", response_class=JSONResponse)
async def job_status_qs(job_id: str): return JSONResponse(engine.load_job(job_id) or {"error": "not found"})

# --- Shadow Memory ---

@router.post("/_shadow_selftest", response_class=JSONResponse)
async def shadow_selftest(request: Request):
    """Proves ShadowStore stage/diff/accept/reject/rollback independent of git or any AI call.
    Writes into a scratch folder under data/ai_manager/_selftest so it never touches real project files."""
    global BI, FM
    shadow = BI.ShadowStore(fm, root / "_shadow")
    FM.write("note.txt", "original line one\noriginal line two\n")
    entry = shadow.stage("note.txt", "original line one\nCHANGED line two\nnew line three\n", author="selftest")
    diff_before_accept = shadow.diff("note.txt")
    accepted = shadow.accept("note.txt")
    final_content = fm.read("note.txt")
    history = shadow.history("note.txt")
    return JSONResponse({"staged_status": entry["status"], "diff": diff_before_accept, "accepted": accepted, "final_file_content": final_content, "history_timestamps": history})