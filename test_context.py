"""Context memory on a stand-in folder: pieces by structure, status rules, budget packing with citations, updates after edits, removal, and the agent knowledge hook."""
import sys, os, json, time, tempfile, asyncio, types
from pathlib import Path
os.chdir(tempfile.mkdtemp())
from tools.ai_manager import engine
from tools.ai_manager import context as C
FAILS = []
def check(name, ok, info=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  {str(info)[:500]}"))
    if not ok: FAILS.append(name)
w = Path("world"); (w / "drafts").mkdir(parents=True); (w / "archive").mkdir()
(w / "physics.md").write_text("# Physics of the world\n\n## Gate travel\nGates cannot open within a gravity well deeper than the moon's. Crossing costs the traveller memory of the last day.\n\n## Light\nThe twin suns keep one hemisphere in daylight.\n")
(w / "drafts" / "gate_idea.md").write_text("# Idea\nMaybe gates could open anywhere if the mage is strong enough.\n")
(w / "archive" / "old_gates.md").write_text("# Old rules\nGates open anywhere, at any time.\n")
(w / "tool.py").write_text("# --- Helpers ---\ndef gate_cost(depth):\n    '''Memory lost per gate crossing.'''\n    return depth * 2\n\nclass Gate:\n    pass\n")
C.set_instance("world", [str(w)], notes="a test world", budget=400)
ctx = C.Context("world")
u = ctx.update()
check("an instance indexes its folder", u["files"] == 4 and u["read"] == 4, u)
r = ctx.query("gates open gravity")
check("canon-like current text comes first, with path, lines, status and heading trail", r.startswith("[world/physics.md:") and "| current |" in r and "Physics of the world > Gate travel" in r, r)
check("drafts come after, marked as drafts", "| draft |" in r and r.index("| draft |") > r.index("| current |"), r)
check("superseded material is left out unless asked for", "old_gates" not in r and "old_gates" in ctx.query("gates open", include="all"), r)
check("code pieces follow its definitions and sections", "Helpers > gate_cost" in ctx.query("gate_cost memory"), ctx.query("gate_cost memory"))
small = ctx.query("gates", budget=60)
check("the budget is kept", sum(len(p) for p in small.split("\n\n")) // 4 < 120 and small.count("[world/") >= 1, small)
time.sleep(0.01)
(w / "physics.md").write_text("# Physics of the world\n\n## Gate travel\nGates open only at dawn.\n")
u = ctx.update()
r = ctx.query("gates")
check("an edit replaces only that file's pieces", u["read"] == 1 and "dawn" in r and "gravity well" not in r, (u, r))
ctx.set_status("world/drafts/*", "abandoned")
check("an owner rule changes what comes back", "mage" not in ctx.query("gates mage"), ctx.query("gates mage"))
ctx.set_status("world/drafts/*", "canon")
r = ctx.query("gates mage")
check("newest rule wins: canon is ranked first", r.startswith("[world/drafts/gate_idea.md") and "| canon |" in r, r)
(w / "tool.py").unlink()
check("a removed file leaves the index", C.Context("world").update()["removed"] == 1 and "gate_cost" not in ctx.query("gate_cost"))
check("nothing found says what was looked for", "nothing in world matches" in ctx.query("zebra"))
from tools.ai_manager import agent_loop as A
cfg = {"knowledge_conn_ids": ["ctx:world"]}
kt = A.KnowledgeTools(cfg, None)
ok, text = asyncio.run(kt.call("knowledge_search", {"query": "gates dawn"}))
check("an agent's knowledge_search reaches the instance", ok and "dawn" in text, text)
check("the instance is offered in the agent's knowledge choices", any(o[0] == "ctx:world" and "context memory" in o[1] for o in A._knowledge_conn_options()), A._knowledge_conn_options())
# --- Meaning search, through stand-in connections ---
import re as _re, zlib
from tools.ai_manager import connections
SYN = {"gate": 0, "gates": 0, "portal": 0, "portals": 0, "doorway": 0, "sun": 1, "suns": 1, "daylight": 1, "dawn": 1, "memory": 2, "forget": 2, "remember": 2}
def vec(t):
    v = [0.0] * 64
    for w in _re.findall(r"\w+", t.lower()): v[SYN[w] if w in SYN else 8 + zlib.crc32(w.encode()) % 56] += 1
    return v
calls = {"embed": 0, "rerank": 0}
async def fake_call(conn, cap, payload, timeout_s=None):
    calls[cap] += 1
    if cap == "embed": return {"embeddings": [vec(t) for t in payload["input"]]}
    q = set(_re.findall(r"\w+", payload["query"].lower()))
    return {"results": [{"index": i, "relevance_score": len(q & set(_re.findall(r"\w+", d.lower())))} for i, d in enumerate(payload["documents"])]}
connections.call_capability = fake_call
connections.get_conn = lambda i: {"_id": i, "connection_type": "magpie"} if i == "node1" else None
(w / "travel.md").write_text("# Travel\n\n## Portals\nA portal opens only for those who forget their way home.\n")
C.set_instance("world", [str(w)], notes="a test world", budget=400, embed={"conn": "node1", "model": "bge-m3"}, rerank={"conn": "node1", "model": "bge-reranker"})
ctx = C.Context("world")
r = asyncio.run(ctx.aquery("doorway rules"))
check("meaning search finds notes worded differently (doorway -> gates, portals)", "Gate travel" in r and "Portals" in r, r)
n0 = calls["embed"]
asyncio.run(ctx.aquery("doorway rules"))
check("pieces are embedded once; a later query only embeds the question", calls["embed"] == n0 + 1, (n0, calls))
check("the reranker reorders the best few", calls["rerank"] >= 2)
(w / "travel.md").write_text("# Travel\n\n## Portals\nA portal opens only for those who forget their way home.\n\n## Rivers\nRivers run uphill near the old gates.\n")
n1 = calls["embed"]
asyncio.run(ctx.aquery("doorway"))
check("an edit embeds only the changed pieces", calls["embed"] - n1 <= 2, calls["embed"] - n1)
C.set_instance("world", [str(w)], notes="a test world", budget=600, embed={"conn": "node1", "model": "bge-m3"})
C.Context("world").set_status("world/drafts/*", "draft")
kt = A.KnowledgeTools({"knowledge_conn_ids": ["ctx:world"], "knowledge_check": True}, None)
names = [t["function"]["name"] if "function" in t else t.get("name") for t in asyncio.run(kt.tools())]
check("knowledge_check is offered only when the run asks for it", "knowledge_check" in names and "knowledge_check" not in [t["function"]["name"] if "function" in t else t.get("name") for t in asyncio.run(A.KnowledgeTools({"knowledge_conn_ids": ["ctx:world"]}, None).tools())], names)
ok, text = asyncio.run(kt.call("knowledge_check", {"proposal": "Let the mage open a portal anywhere, even at night"}))
check("a proposal is checked against settled material only: the rule comes back, the draft does not", ok and "Settled material" in text and "dawn" in text and "| draft |" not in text, text)
check("the run is told to say what it checked", "knowledge_check" in asyncio.run(kt.open()))
C.set_instance("world", [str(w)], budget=400, embed={"conn": "gone", "model": "x"})
try: asyncio.run(C.Context("world").aquery("x")); check("a missing connection is reported", False)
except RuntimeError as e: check("a missing connection is reported", "gone" in str(e), e)
print("ALL PASSED" if not FAILS else f"FAILED: {FAILS}")
