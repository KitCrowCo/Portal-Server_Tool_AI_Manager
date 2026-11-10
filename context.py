"""context.py - context memory: the pieces of a body of material an agent needs, current, cited and within a token budget. One instance per domain (a codebase, a world's notes, household papers, work); nothing is shared between instances.
Built from the files themselves, with no model: pieces follow the material's own shape (Markdown and text by heading, code by top-level definition and "# --- Section ---" header, PDF by page), each keeping its path, lines, heading trail, date and status.
Status is the owner's: data/_tools/ai_manager/context/<instance>/status.json is a list of [glob, status] rules, later rules winning (canon, current, draft, alternate, superseded, abandoned); unmatched files are current. Superseded and abandoned pieces are only returned when asked for.
Instances are listed in data/_tools/ai_manager/context/instances.json: {name: {"folders": [...], "notes": "what it holds", "budget": tokens, "exclude": [globs]}}. update() re-reads only files whose size or time changed, so it is cheap before every query.
Meaning search (optional): "embed": {"conn", "model"} in an instance has each piece embedded once (cached by its text and the model) through a connection declaring "embed" (a Magpie node); "rerank": {"conn", "model"} reorders the best few through one declaring "rerank".
Word and meaning results are merged by reciprocal rank; pieces not embedded yet are still found by their words, and each query embeds at most EMBED_PER_QUERY new pieces (embed_all() does the rest).
Claims (optional): "claims": {"conn", "model"} has a chat model list what each piece states (subject, attribute, value, the words it used), once per piece. Claims are never merged: two current or canon sources saying different things about the same subject and attribute make a conflict, which the owner settles (decisions.jsonl, appended only); a settled answer is put first in every search that touches it.
Agents reach an instance as knowledge group "ctx:<name>" (agent_loop.KnowledgeTools). Nothing here contacts anything but the connections an instance names."""
import os, re, json, time, fnmatch, sqlite3, hashlib
from pathlib import Path
import numpy as np
from tools.ai_manager import code_index, connections

ROOT = Path("./data/_tools/ai_manager/context")
STATUS = ("canon", "current", "draft", "alternate", "superseded", "abandoned")
WEIGHT = {"canon": 1.3, "current": 1.0, "draft": 0.6, "alternate": 0.5, "superseded": 0.25, "abandoned": 0.2}
DEFAULT_RULES = [["*/drafts/*", "draft"], ["drafts/*", "draft"], ["*/archive/*", "superseded"], ["archive/*", "superseded"], ["*/old/*", "superseded"], ["old/*", "superseded"]]
TEXT = {".md", ".txt", ".rst", ".org", ".py", ".js", ".mjs", ".html", ".htm", ".css", ".json", ".yml", ".yaml", ".toml", ".sh", ".c", ".cpp", ".h", ".csv", ".tex"}
MAX_LINES, MAX_BYTES = 80, 3_000_000
EMBED_PER_QUERY, EMBED_BATCH, RERANK_TOP, RRF_K = 64, 16, 40, 60
HIDE = {"current": ("superseded", "abandoned"), "all": (), "binding": ("draft", "alternate", "superseded", "abandoned")}   # include= in a query: what is left out
_VERSION = "2"   # index layout: an index made by another version is rebuilt from the folders
_MATRIX: dict = {}   # instance -> (stamp, piece ids, unit vectors), reloaded when the stored vectors change
_HEAD = re.compile(r"^(#{1,6})\s+(.+)")
_SCHEMA = """CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS pieces (id INTEGER PRIMARY KEY, path TEXT, a INTEGER, b INTEGER, trail TEXT, body TEXT, mtime REAL, tokens INTEGER, key TEXT);
CREATE INDEX IF NOT EXISTS piece_path ON pieces(path); CREATE INDEX IF NOT EXISTS piece_key ON pieces(key);
CREATE TABLE IF NOT EXISTS emb (key TEXT, model TEXT, v BLOB, PRIMARY KEY (key, model));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(trail, body, path UNINDEXED, content='pieces', content_rowid='id');
CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY, key TEXT, model TEXT, subject TEXT, attribute TEXT, value TEXT, quote TEXT, s TEXT, a TEXT);
CREATE INDEX IF NOT EXISTS claim_key ON claims(key); CREATE INDEX IF NOT EXISTS claim_sa ON claims(s, a);
CREATE TABLE IF NOT EXISTS claims_done (key TEXT, model TEXT, PRIMARY KEY (key, model));"""
CLAIM_PROMPT = """You list the claims a passage makes, so they can be checked against other passages later. Answer with one JSON object per line and nothing else:
{"subject": "the thing it is about, by name", "attribute": "what about it, a few lowercase words", "value": "what the passage says", "quote": "the passage's own words"}
Only what the passage states, not guesses or summaries. Rules, limits and numbers matter most ("cannot", "only", "always", amounts, dates). At most 12 lines. If it states nothing checkable, answer NONE."""

# --- Instances ---

def instances() -> dict:
    f = ROOT / "instances.json"
    return json.loads(f.read_text()) if f.exists() else {}

def set_instance(name: str, folders: list, notes: str = "", budget: int = 6000, exclude: list = None, embed: dict = None, rerank: dict = None, claims: dict = None):
    """Adds or changes an instance's settings; its index follows the folders on the next update. embed / rerank / claims: {"conn": connection id, "model": name}, or None to go without."""
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "instances.json").write_text(json.dumps({**instances(), name: {"folders": folders, "notes": notes, "budget": int(budget), "exclude": exclude or [], **{k: v for k, v in (("embed", embed), ("rerank", rerank), ("claims", claims)) if v}}}, indent=1))

def remove_instance(name: str) -> str:
    """Takes an instance out of instances.json and moves its folder (index, status rules) to context/_trash/; the material itself is untouched. -> where it went."""
    d, inst = ROOT / re.sub(r"\W+", "_", name), instances()
    inst.pop(name, None)
    (ROOT / "instances.json").write_text(json.dumps(inst, indent=1))
    to = ROOT / "_trash" / f"""{d.name}.{time.strftime("%Y%m%d-%H%M%S")}"""
    if d.exists(): to.parent.mkdir(parents=True, exist_ok=True); d.rename(to)
    return str(to)

def conn(name: str) -> dict:
    """An instance as agent_loop sees a knowledge group: a connection-shaped dict of type "context"."""
    return {"_id": f"ctx:{name}", "connection_type": "context", "display_name": name, "values": {"domain_notes": instances()[name].get("notes", "")}}

# --- Pieces ---

def _norm(s: str) -> str: return " ".join(re.findall(r"\w+", str(s).lower()))

def _claim(line: str):
    """One claim from a line of the model's answer, or None when the line is not a usable claim."""
    m = re.search(r"\{.*\}", line)
    if not m: return None
    try: x = json.loads(m.group(0))
    except json.JSONDecodeError: return None
    return x if isinstance(x, dict) and all(str(x.get(k, "")).strip() for k in ("subject", "attribute", "value")) else None

def _trail(stack: list) -> str: return " > ".join(t for _, t in stack)

def pieces(rel: str, text: str) -> list:
    """[(first line, last line, heading trail, text)] for one file: split where its own structure starts something new, long stretches split again at blank lines."""
    lines = text.splitlines()
    if not lines: return []
    starts, ext = {1: ""}, Path(rel).suffix.lower()
    if ext in (".md", ".txt", ".rst", ".org", ".tex"):
        stack = []
        for i, l in enumerate(lines, 1):
            if (m := _HEAD.match(l)):
                stack = [(d, t) for d, t in stack if d < len(m.group(1))] + [(len(m.group(1)), m.group(2).strip())]
                starts[i] = _trail(stack)
    else:
        section = ""
        syms = sorted(code_index.parse(rel, text)[1], key=lambda s: s[2])
        for name, kind, line, end, parent, sig, doc in syms:
            if kind == "section": section = name
            if kind == "section" or (kind in ("function", "class") and not parent): starts[line] = f"""{section + " > " if section and kind != "section" else ""}{name}"""
    marks = sorted(starts)
    out = []
    for k, a in enumerate(marks):
        b = (marks[k + 1] - 1) if k + 1 < len(marks) else len(lines)
        trail = starts[a]
        while a <= b:
            end = min(b, a + MAX_LINES - 1)
            if end < b:
                gap = next((j for j in range(end, a + MAX_LINES // 3, -1) if not lines[j - 1].strip()), end)   # end a long stretch at a blank line when there is one
                end = gap
            body = "\n".join(lines[a - 1:end]).strip()
            if body and not all(_HEAD.match(x) or not x.strip() for x in body.splitlines()): out.append((a, end, trail, body))   # a heading with nothing under it is carried by the next piece's trail
            a = end + 1
    return out

def _pdf_pieces(path: Path) -> list:
    from pypdf import PdfReader
    return [(i, i, f"page {i}", t.strip()) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

# --- Index ---

class Context:
    def __init__(self, name: str):
        self.name, self.cfg = name, instances()[name]
        self.dir = ROOT / re.sub(r"\W+", "_", name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.roots = [Path(os.path.expanduser(f)).resolve() for f in self.cfg["folders"]]

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.dir / "index.db")
        c.row_factory = sqlite3.Row
        old = c.execute("SELECT name FROM sqlite_master WHERE name='meta'").fetchone() is None or (c.execute("SELECT v FROM meta WHERE k='version'").fetchone() or [None])[0] != _VERSION
        if old:
            for t in ("search", "pieces", "files", "meta"): c.execute(f"DROP TABLE IF EXISTS {t}")   # embeddings and claims are kept: they are keyed by text, not by layout
        if old or c.execute("SELECT name FROM sqlite_master WHERE name='claims_done'").fetchone() is None:
            c.executescript(_SCHEMA)   # every statement is IF NOT EXISTS, so this also adds tables newer than an existing index
            if old: c.execute("INSERT INTO meta VALUES ('version', ?)", (_VERSION,))
            c.commit()
        return c

    def rules(self) -> list:
        f = self.dir / "status.json"
        return DEFAULT_RULES + (json.loads(f.read_text()) if f.exists() else [])

    def status_of(self, rel: str, rules: list) -> str: return next((s for g, s in reversed(rules) if fnmatch.fnmatch(rel, g)), "current")

    def own_rules(self) -> list:
        f = self.dir / "status.json"
        return json.loads(f.read_text()) if f.exists() else []

    def set_rules(self, rules: list):
        """Replaces the owner's status rules (the page edits them as a list); the previous list is kept in status_history.jsonl."""
        if any(st not in STATUS for _, st in rules): raise ValueError(f"status must be one of {STATUS}")
        with open(self.dir / "status_history.jsonl", "a") as h: h.write(json.dumps({"t": time.time(), "before": self.own_rules()}) + "\n")
        (self.dir / "status.json").write_text(json.dumps(rules, indent=1))

    def stats(self) -> dict:
        c = self._conn()
        model = (self.cfg.get("embed") or {}).get("model", "")
        out = {"files": c.execute("SELECT count(*) FROM files").fetchone()[0], "pieces": c.execute("SELECT count(*) FROM pieces").fetchone()[0], "tokens": c.execute("SELECT coalesce(sum(tokens), 0) FROM pieces").fetchone()[0],
               "embedded": c.execute("SELECT count(*) FROM pieces p JOIN emb e ON e.key = p.key AND e.model = ?", (model,)).fetchone()[0] if model else None, "by_status": dict(c.execute("SELECT status, count(*) FROM files GROUP BY status").fetchall()),
               "claimed": c.execute("SELECT count(DISTINCT p.key) FROM pieces p JOIN claims_done d ON d.key = p.key AND d.model = ?", ((self.cfg.get("claims") or {}).get("model", ""),)).fetchone()[0] if self.cfg.get("claims") else None}
        c.close()
        return out

    def set_status(self, glob: str, status: str):
        """Adds an owner rule (newest wins); the next update applies it."""
        if status not in STATUS: raise ValueError(f"status must be one of {STATUS}")
        f = self.dir / "status.json"
        f.write_text(json.dumps((json.loads(f.read_text()) if f.exists() else []) + [[glob, status]], indent=1))

    def _walk(self):
        """(name shown to agents, absolute path): a folder's files under its own name, so two folders never collide."""
        ex = self.cfg.get("exclude") or []
        for root in self.roots:
            for d, dirs, files in os.walk(root):
                base = Path(d).relative_to(root)
                dirs[:] = sorted(x for x in dirs if not x.startswith(".") and not any(fnmatch.fnmatch((base / x).as_posix(), g) for g in ex))
                for f in sorted(files):
                    rel = (Path(root.name) / base / f).as_posix()
                    if Path(f).suffix.lower() in TEXT | {".pdf"} and not any(fnmatch.fnmatch(rel, g) for g in ex): yield rel, Path(d) / f

    def update(self) -> dict:
        """New and changed files re-read, removed ones dropped, status rules re-applied. -> counts and seconds."""
        t0, c, rules, read = time.time(), self._conn(), self.rules(), 0
        known = {r["path"]: (r["mtime"], r["size"], r["status"]) for r in c.execute("SELECT path, mtime, size, status FROM files")}
        seen = set()
        for rel, p in self._walk():
            seen.add(rel)
            st, status = p.stat(), self.status_of(rel, rules)
            if known.get(rel, (None, None, None))[:2] == (st.st_mtime, st.st_size):
                if known[rel][2] != status: c.execute("UPDATE files SET status=? WHERE path=?", (status, rel))
                continue
            self._drop(c, rel)
            ps = _pdf_pieces(p) if p.suffix.lower() == ".pdf" else pieces(rel, p.read_text(errors="replace")) if st.st_size <= MAX_BYTES else []
            c.execute("INSERT INTO files VALUES (?,?,?,?)", (rel, st.st_mtime, st.st_size, status))
            for a, b, trail, body in ps:
                cur = c.execute("INSERT INTO pieces (path, a, b, trail, body, mtime, tokens, key) VALUES (?,?,?,?,?,?,?,?)", (rel, a, b, trail, body, st.st_mtime, len(body) // 4 + 8, hashlib.sha1(f"{trail}\n{body}".encode()).hexdigest()))
                c.execute("INSERT INTO search (rowid, trail, body, path) VALUES (?,?,?,?)", (cur.lastrowid, trail, body, rel))
            read += 1
        for rel in known.keys() - seen: self._drop(c, rel)
        c.commit()
        c.close()
        return {"files": len(seen), "read": read, "removed": len(known.keys() - seen), "seconds": round(time.time() - t0, 2)}

    def _drop(self, c, rel: str):
        for r in c.execute("SELECT id, trail, body, path FROM pieces WHERE path=?", (rel,)).fetchall(): c.execute("INSERT INTO search (search, rowid, trail, body, path) VALUES ('delete', ?, ?, ?, ?)", (r["id"], r["trail"], r["body"], r["path"]))
        c.execute("DELETE FROM pieces WHERE path=?", (rel,))
        c.execute("DELETE FROM files WHERE path=?", (rel,))

    # --- Meaning search ---

    def _econn(self, kind: str) -> tuple:
        e = self.cfg.get(kind) or {}
        c = connections.get_conn(e.get("conn", "")) if e.get("conn") else None
        if e and not c: raise RuntimeError(f"""context {self.name}: the {kind} connection {e.get('conn')} no longer exists - fix the instance's settings""")
        return c, e.get("model", "")

    async def embed_pending(self, limit: int = EMBED_PER_QUERY) -> int:
        """Embeds up to limit pieces that have no vector for the instance's model yet. -> how many were embedded."""
        conn, model = self._econn("embed")
        if not conn: return 0
        c = self._conn()
        rows = c.execute("SELECT DISTINCT p.key, p.trail, p.body FROM pieces p LEFT JOIN emb e ON e.key = p.key AND e.model = ? WHERE e.key IS NULL LIMIT ?", (model, limit)).fetchall()
        done = 0
        for k in range(0, len(rows), EMBED_BATCH):
            batch = rows[k:k + EMBED_BATCH]
            r = await connections.call_capability(conn, "embed", {"model": model, "input": [f"""{x["trail"]}\n{x["body"]}"""[:6000] for x in batch]})
            if r.get("error"): raise RuntimeError(f"""context {self.name}: embedding failed: {r['error']}""")
            c.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?)", [(x["key"], model, np.asarray(v, dtype=np.float32).tobytes()) for x, v in zip(batch, r["embeddings"])])
            c.commit()
            done += len(batch)
        c.close()
        return done

    async def embed_all(self) -> int:
        """Embeds every piece still missing a vector, in batches. -> how many."""
        total = 0
        while (n := await self.embed_pending(512)): total += n
        return total

    def _matrix(self, model: str) -> tuple:
        c = self._conn()
        stamp = (c.execute("SELECT count(*) FROM emb WHERE model=?", (model,)).fetchone()[0], c.execute("SELECT count(*), max(id) FROM pieces").fetchone()[:])
        if _MATRIX.get(self.name, (None,))[0] != stamp:
            rows = c.execute("SELECT p.id, e.v FROM pieces p JOIN emb e ON e.key = p.key AND e.model = ?", (model,)).fetchall()
            m = np.vstack([np.frombuffer(r["v"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), dtype=np.float32)
            _MATRIX[self.name] = (stamp, np.array([r["id"] for r in rows]), m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9))
        c.close()
        return _MATRIX[self.name][1:]

    async def _meaning(self, text: str, n: int = 200) -> list:
        """Piece ids nearest in meaning to text, best first."""
        conn, model = self._econn("embed")
        if not conn: return []
        await self.embed_pending()
        ids, m = self._matrix(model)
        if not len(ids): return []
        r = await connections.call_capability(conn, "embed", {"model": model, "input": [text]})
        if r.get("error"): raise RuntimeError(f"""context {self.name}: embedding the question failed: {r['error']}""")
        q = np.asarray(r["embeddings"][0], dtype=np.float32)
        sims = m @ (q / max(float(np.linalg.norm(q)), 1e-9))
        return ids[np.argsort(-sims)[:n]].tolist()

    async def _rerank(self, text: str, rows: list) -> list:
        conn, model = self._econn("rerank")
        if not conn or len(rows) < 2: return rows
        top = rows[:RERANK_TOP]
        r = await connections.call_capability(conn, "rerank", {"model": model, "query": text, "documents": [f"""{x["trail"]}\n{x["body"]}"""[:4000] for x in top]})
        if r.get("error"): raise RuntimeError(f"""context {self.name}: reranking failed: {r['error']}""")
        score = {int(x["index"]): float(x.get("relevance_score", x.get("score", 0))) for x in r["results"]}
        return sorted(top, key=lambda x: -score.get(top.index(x), -1e9) * WEIGHT.get(x["status"], 1.0)) + rows[RERANK_TOP:]

    # --- Claims and conflicts ---

    async def extract_pending(self, limit: int = 32) -> int:
        """Lists the claims of up to limit pieces not read yet by the instance's claims model. -> how many pieces were read."""
        e = self.cfg.get("claims") or {}
        if not e: return 0
        conn = connections.get_conn(e.get("conn", ""))
        if not conn: raise RuntimeError(f"""context {self.name}: the claims connection {e.get('conn')} no longer exists - fix the instance's settings""")
        c = self._conn()
        rows = c.execute("SELECT DISTINCT p.key, p.trail, p.body FROM pieces p LEFT JOIN claims_done d ON d.key = p.key AND d.model = ? WHERE d.key IS NULL LIMIT ?", (e["model"], limit)).fetchall()
        for r in rows:
            text = "".join([t async for t, _ in connections.stream_llm(conn, [{"role": "system", "content": CLAIM_PROMPT}, {"role": "user", "content": f"""[{r["trail"]}]\n{r["body"]}"""[:8000]}], e["model"], think=False)])
            got = [x for x in (_claim(ln) for ln in text.splitlines()) if x]
            c.executemany("INSERT INTO claims (key, model, subject, attribute, value, quote, s, a) VALUES (?,?,?,?,?,?,?,?)", [(r["key"], e["model"], x["subject"], x["attribute"], x["value"], x.get("quote", ""), _norm(x["subject"]), _norm(x["attribute"])) for x in got])
            c.execute("INSERT OR REPLACE INTO claims_done VALUES (?, ?)", (r["key"], e["model"]))
            c.commit()
        c.close()
        return len(rows)

    async def extract_all(self) -> int:
        """Reads every piece not read yet, in batches. -> how many."""
        total = 0
        while (n := await self.extract_pending(64)): total += n
        return total

    def decisions(self) -> list:
        f = self.dir / "decisions.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []

    def conflicts(self) -> list:
        """Subjects and attributes on which current or canon sources disagree and no decision covers every value now seen: [{"id", "subject", "attribute", "claims": [{"value", "quote", "path", "a", "b", "status"}]}]."""
        model = (self.cfg.get("claims") or {}).get("model", "")
        c = self._conn()
        rows = c.execute("SELECT DISTINCT c.subject, c.attribute, c.value, c.quote, c.s, c.a, p.path, p.a AS la, p.b AS lb, f.status FROM claims c JOIN pieces p ON p.key = c.key JOIN files f ON f.path = p.path WHERE c.model = ? AND f.status IN ('canon', 'current')", (model,)).fetchall()
        c.close()
        groups, settled = {}, {(d["s"], d["a"]): set(d["values"]) for d in self.decisions()}
        for r in rows: groups.setdefault((r["s"], r["a"]), []).append(r)
        out = []
        for (sn, an), rs in groups.items():
            vals = {_norm(r["value"]) for r in rs}
            if len(vals) < 2 or all(any(set(v.split()) <= set(w.split()) or set(w.split()) <= set(v.split()) for w in vals - {v}) for v in vals) or vals <= settled.get((sn, an), set()): continue   # one value, values that only restate each other, or already settled
            out.append({"id": hashlib.sha1(f"{sn}|{an}".encode()).hexdigest()[:10], "subject": rs[0]["subject"], "attribute": rs[0]["attribute"], "s": sn, "a": an, "claims": [{"value": r["value"], "quote": r["quote"], "path": r["path"], "a": r["la"], "b": r["lb"], "status": r["status"]} for r in rs]})
        return out

    def decide(self, cid: str, value: str, note: str = "") -> dict:
        """The owner settles a conflict: value is what holds (one of the claims' values, or new words), or "contexts" when the claims are about different times or versions and both stand. Appended to decisions.jsonl; the material is not changed."""
        k = next((x for x in self.conflicts() if x["id"] == cid), None)
        if not k: raise ValueError("that conflict is no longer open (settled, or its sources changed)")
        d = {"t": time.time(), "subject": k["subject"], "attribute": k["attribute"], "s": k["s"], "a": k["a"], "values": sorted({_norm(x["value"]) for x in k["claims"]}), "value": value, "note": note}
        with open(self.dir / "decisions.jsonl", "a") as f: f.write(json.dumps(d) + "\n")
        return d

    def settled(self, words: list) -> list:
        """Decisions whose subject or attribute shares a word with the search, the newest per subject and attribute."""
        ws = {_norm(w) for w in words if len(w) > 2}
        latest = {}
        for d in self.decisions():
            if ws & set(d["s"].split()) or ws & set(d["a"].split()): latest[(d["s"], d["a"])] = d
        return list(latest.values())

    # --- Query ---

    def _words(self, c, words: list, hide: tuple) -> list:
        return c.execute(f"""SELECT p.*, f.status, bm25(search, 2.0, 1.0) AS score FROM search JOIN pieces p ON p.id = search.rowid JOIN files f ON f.path = p.path WHERE search MATCH ? {"AND f.status NOT IN (" + ",".join("?" * len(hide)) + ")" if hide else ""} ORDER BY score LIMIT 200""", (" OR ".join(f'"{w}"' for w in words), *hide)).fetchall()   # any of the words; bm25 ranks pieces with more of them, and rarer ones, first

    def _pack(self, ranked: list, budget: int, words: list) -> str:
        out = [f"""[settled by the owner {time.strftime("%Y-%m-%d", time.localtime(d["t"]))}] {d["subject"]} - {d["attribute"]}: {"both stand, in different contexts" if d["value"] == "contexts" else d["value"]}{" (" + d["note"] + ")" if d.get("note") else ""}""" for d in self.settled(words)]
        used, taken = sum(len(x) // 4 for x in out), {}
        for r in ranked:
            if any(not (r["b"] < a or r["a"] > b) for a, b in taken.get(r["path"], [])): continue
            if used + r["tokens"] > budget: continue
            taken.setdefault(r["path"], []).append((r["a"], r["b"]))
            used += r["tokens"]
            out.append(f"""[{r["path"]}:{r["a"]}{"-" + str(r["b"]) if r["b"] != r["a"] else ""} | {r["status"]} | {time.strftime("%Y-%m-%d", time.localtime(r["mtime"]))}{" | " + r["trail"] if r["trail"] else ""}]\n{r["body"]}""")
        return "\n\n".join(out) if out else f"""(nothing in {self.name} matches: {" ".join(words)})"""

    def query(self, text: str, budget: int = None, include: str = "current") -> str:
        """Word search only: the best pieces for text, packed into budget tokens, each headed [path:lines | status | date | trail]. include: "current" leaves out superseded and abandoned material, "binding" also drafts and alternates (what a proposal must respect), "all" keeps everything (ranked by status)."""
        self.update()
        words = re.findall(r"\w+", text.replace("_", " "))
        if not words: return "give some words to search for"
        c = self._conn()
        rows = self._words(c, words, HIDE[include])
        c.close()
        return self._pack(sorted(rows, key=lambda r: (r["score"] * WEIGHT.get(r["status"], 1.0), -r["mtime"])), int(budget or self.cfg.get("budget", 6000)), words)   # bm25 is negative: lower is better, so a weight above 1 lifts a piece

    async def aquery(self, text: str, budget: int = None, include: str = "current") -> str:
        """Word and meaning search merged by reciprocal rank (status-weighted), reranked when the instance has a reranker, then packed. Without "embed" it is query()."""
        import asyncio
        if not self.cfg.get("embed"): return await asyncio.to_thread(self.query, text, budget, include)
        await asyncio.to_thread(self.update)
        words = re.findall(r"\w+", text.replace("_", " ")) or [text]
        hide = HIDE[include]
        near = await self._meaning(text)
        c = self._conn()
        wrows = self._words(c, words, hide) if re.findall(r"\w+", text) else []
        extra = [i for i in near if i not in {r["id"] for r in wrows}]
        mrows = {r["id"]: r for r in c.execute(f"""SELECT p.*, f.status, 0 AS score FROM pieces p JOIN files f ON f.path = p.path WHERE p.id IN ({",".join("?" * len(extra))})""", extra).fetchall()} if extra else {}
        c.close()
        rows = {r["id"]: r for r in wrows} | {i: r for i, r in mrows.items() if r["status"] not in hide}
        fused = {i: 0.0 for i in rows}
        for rank, r in enumerate(wrows): fused[r["id"]] += 1 / (RRF_K + rank)
        for rank, i in enumerate(near):
            if i in fused: fused[i] += 1 / (RRF_K + rank)
        ranked = sorted(rows.values(), key=lambda r: (-fused[r["id"]] * WEIGHT.get(r["status"], 1.0), -r["mtime"]))
        return self._pack(await self._rerank(text, ranked), int(budget or self.cfg.get("budget", 6000)), words)
