#tools/ai_manager
"""
ai_manager — sole centralized tool for AI operations. Owns connections, the step registry,
and pipeline execution. modules/ai_tools/* (Athena, Kimi, Tessa, Image) are UI surfaces that
call into this tool directly (ENV["tools"]["ai_manager"]); they never hold their own
connections or execution loops - avoids the race conditions of tools calling other tools.
"""
import sys, os
from pathlib import Path
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import engine
import steps
# from engine import *
# from steps import *
from connections import list_conns, get_conn

TOOL_META = {"label": "AI Manager", "icon": "&#x1F9E0;", "description": "Centralized AI connections, steps, and pipeline execution"}
router = APIRouter()
ENV: dict = {}
_P = "/tool/ai_manager"

def init_module(env: dict):
    global ENV
    ENV.update(env)
    engine.init(env)
    steps.register_builtins()
    print(f"[ai_manager] ready | step types: {[s['type'] for s in steps.list_step_types()]}")

def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

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

# -- Simple Chat (working demonstration: chat with an optional attached knowledge base) --
# This is intentionally minimal - a 1-or-2-node inline Flow, not a saved pipeline, not routed through Tessa/Kimi UI.
# Proves the "just chat with knowledge as needed" path works without waiting on the full builder migration.

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    conn_opts = "".join(f'<option value="{c["_id"]}">{_esc(c.get("display_name",c["_id"]))}</option>' for c in list_conns())
    kg_opts = '<option value="">(no knowledge base)</option>' + "".join(f'<option value="{c["_id"]}">{_esc(c.get("display_name",c["_id"]))}</option>' for c in list_conns(conn_type="lightrag"))
    return HTMLResponse(f"""<div style="max-width:44rem;margin:0 auto;padding:1.5rem;display:flex;flex-direction:column;gap:.6rem;height:100%;box-sizing:border-box">
        <div style="display:flex;gap:.5rem">
            <select id="chat-conn" class="module-select" style="flex:1;margin:0">{conn_opts}</select>
            <select id="chat-kg" class="module-select" style="flex:1;margin:0">{kg_opts}</select>
        </div>
        <div id="chat-log" style="flex:1;overflow-y:auto;border:var(--border-thick) solid var(--border);border-radius:var(--radius);padding:.6rem;display:flex;flex-direction:column;gap:.5rem"></div>
        <div id="chat-stream" style="font-size:.85rem;color:var(--text_muted);white-space:pre-wrap"></div>
        <form id="chat-form" style="display:flex;gap:.5rem" onsubmit="return false">
            <input id="chat-input" type="text" class="module-select" style="flex:1;margin:0" placeholder="Ask something...">
            <button class="ui-btn" onclick="aimChatSend()">Send</button>
        </form>
        <script>
            function aimChatSend(){{
                var input = document.getElementById('chat-input'); var text = input.value.trim(); if(!text) return;
                var log = document.getElementById('chat-log');
                log.insertAdjacentHTML('beforeend', '<div style="align-self:flex-end;background:var(--accent_dim);padding:.4rem .6rem;border-radius:var(--radius);max-width:80%">'+text+'</div>');
                input.value=''; document.getElementById('chat-stream').textContent='';
                fetch('{_P}/chat/send', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body:new URLSearchParams({{conn_id:document.getElementById('chat-conn').value, kg_id:document.getElementById('chat-kg').value, text:text}})}}).then(r=>r.json()).then(d=>{{ if(d.error){{ document.getElementById('chat-stream').textContent = 'Error: '+d.error; return; }} window._aimJob = d.job_id; }});
            }}
            document.addEventListener('pipeline:stream', function(e){{ if(e.detail.job_id !== window._aimJob) return; document.getElementById('chat-stream').textContent += e.detail.delta; }});
            document.addEventListener('pipeline:done', function(e){{
                if(e.detail.job_id !== window._aimJob) return;
                fetch('{_P}/job/'+e.detail.job_id).then(r=>r.json()).then(job=>{{
                    var answer = (job.scratch && job.scratch.chat && job.scratch.chat.answer) || '(no answer)';
                    document.getElementById('chat-log').insertAdjacentHTML('beforeend', '<div style="background:var(--glass);padding:.4rem .6rem;border-radius:var(--radius);max-width:80%">'+answer+'</div>');
                    document.getElementById('chat-stream').textContent = '';
                }});
            }});
            document.addEventListener('pipeline:error', function(e){{ if(e.detail.job_id === window._aimJob) document.getElementById('chat-stream').textContent = 'Error: '+(e.detail.message||'unknown'); }});
        </script>
    </div>""")

@router.post("/chat/send", response_class=JSONResponse)
async def chat_send(request: Request, conn_id: str = Form(...), kg_id: str = Form(""), text: str = Form(...)):
    user = request.state.user
    flow = {"nodes": []}
    if kg_id: flow["nodes"].append({"id": "kq", "type": "knowledge_query", "config": {"conn_id": kg_id, "query_template": "{input}", "result_key": "kg_context"}, "next": ["chat"]})
    chat_node = {"id": "chat", "type": "chat", "prev": ["kq"] if kg_id else [], "config": {"conn_id": conn_id,
                                                                                           "system_prompt": "Use the retrieved context if relevant to answer the user's question." if kg_id else "",
                                                                                           "user_template": ("Context:\n{kq.kg_context}\n\nQuestion: {input}" if kg_id else "{input}"),
                                                                                           "result_key": "answer"}}
    flow["nodes"].append(chat_node)
    job_id, err = engine.submit(user.username, kind="inline", inline_flow=flow, inputs={"input": text})
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