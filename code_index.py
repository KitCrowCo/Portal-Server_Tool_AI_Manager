"""code_index.py - a map of the code in one folder, for agents working in a project too large to read: definitions, where names are used, intents and routes, CSS classes, section headers and docstrings, searchable by words.
Built from the files themselves - Python through its own parser (ast), JS / CSS / HTML / Markdown and the HTML inside Python strings through patterns. No model is involved and nothing leaves this server.
Kept per folder in data/ai_manager/code_index/<folder>.db (SQLite with full-text search). update() re-reads only files whose size or time changed, so it is cheap to call before every run.
The portal's own conventions are understood: an intent ("finance_rec_cell") is linked to its handler where IM.scripts registers it and to every place a page sends it (hx-vals "type"), a route to its function, a "# --- Section ---" header to its lines."""
import os, re, ast, time, sqlite3
from pathlib import Path

INDEX_DIR = Path("./data/ai_manager/code_index")
_LANG = {".py": "py", ".js": "js", ".mjs": "js", ".html": "html", ".htm": "html", ".css": "css", ".md": "md", ".json": "json", ".sh": "sh", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".c": "c", ".cpp": "c", ".h": "c", ".txt": "txt"}
_MAX_BYTES = 1_500_000   # larger files (minified bundles, data dumps) are listed but not parsed
_SECTION = re.compile(r"^\s*(?:#|//|/\*)\s*-{3}\s*(.+?)\s*-{3}")
_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+)")
_INTENT_STR = re.compile(r"""["']?type["']?\s*:\s*["']([a-z][a-z0-9]*_[a-z0-9_]+)["']""")   # "type": "finance_rec_cell" in hx-vals, htmx.ajax values, json.dumps dicts
_INTENT_KEY = re.compile(r"^[a-z][a-z0-9]*_[a-z0-9_]+$")
_WORD = re.compile(r"^[A-Za-z_][\w-]{2,60}$")   # identifier-like strings: intent names, keys, short action names
_ROUTE_STR = re.compile(r"""hx-(?:get|post|put|delete)=["'](/[^"'?{ ]+)""")
_CSS_DEF = re.compile(r"(?m)^\s*\.([A-Za-z][\w-]*)[^{\n;]*\{")
_CSS_USE = re.compile(r"""class=["']([^"'{}]+)["']""")
_ID_DEF = re.compile(r"""\bid=["']([A-Za-z][\w-]*)["']""")
_ID_USE = re.compile(r"""(?:hx-target|hx-include)=["'][^"']*#([A-Za-z][\w-]*)""")
_VERSION = "3"   # parser version: an index made by another version is rebuilt
_JS_DEFS = [re.compile(p) for p in (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", r"^\s*(?:window\.)?(\w+)\s*=\s*(?:window\.\w+\s*\|\|\s*)?(?:async\s+)?function\b", r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>")]
_JS_CALL = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
_JS_SKIP = frozenset("if for while switch catch function return typeof new await async with".split())
_SCHEMA = """CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL, size INTEGER, lines INTEGER, lang TEXT, summary TEXT, problem TEXT);
CREATE TABLE IF NOT EXISTS symbols (path TEXT, name TEXT, kind TEXT, line INTEGER, end_line INTEGER, parent TEXT, signature TEXT, doc TEXT);
CREATE TABLE IF NOT EXISTS refs (path TEXT, name TEXT, line INTEGER, kind TEXT);
CREATE INDEX IF NOT EXISTS sym_name ON symbols(name); CREATE INDEX IF NOT EXISTS sym_path ON symbols(path);
CREATE INDEX IF NOT EXISTS ref_name ON refs(name); CREATE INDEX IF NOT EXISTS ref_path ON refs(path);
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(path, name, kind, text, line UNINDEXED);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);"""

def _words(name: str) -> str: return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ")   # rec_cell, recCell and rec-cell all match "rec cell"
def _first(doc: str) -> str: return (doc or "").strip().split("\n")[0][:240]

# --- Parsers: text -> (summary, symbols [(name, kind, line, end, parent, signature, doc)], refs [(name, line, kind)]) ---

def _strings(text: str, line0: int, syms: list, refs: list):
    """What HTML and CSS inside a string (or a template) defines and uses: intents sent, routes posted to, CSS classes defined and used."""
    for rx, kind in ((_INTENT_STR, "intent"), (_ROUTE_STR, "route")):
        for m in rx.finditer(text): refs.append((m.group(1), line0 + text.count("\n", 0, m.start()), kind))
    for m in _CSS_DEF.finditer(text): syms.append((m.group(1), "css", line0 + text.count("\n", 0, m.start()), 0, "", "", ""))
    for m in _ID_DEF.finditer(text): syms.append((m.group(1), "html-id", line0 + text.count("\n", 0, m.start()), 0, "", "", ""))
    for m in _ID_USE.finditer(text): refs.append((m.group(1), line0 + text.count("\n", 0, m.start()), "target"))
    for m in _CSS_USE.finditer(text):
        for c in m.group(1).split(): refs.append((c, line0 + text.count("\n", 0, m.start()), "css"))

def _sections(lines: list, syms: list):
    for i, l in enumerate(lines, 1):
        if (m := _SECTION.match(l)): syms.append((m.group(1), "section", i, 0, "", l.strip(), ""))

def _py(text: str) -> tuple:
    tree, syms, refs = ast.parse(text), [], []
    def sig(n): return f"""{"async " if isinstance(n, ast.AsyncFunctionDef) else ""}def {n.name}({ast.unparse(n.args)}){" -> " + ast.unparse(n.returns) if n.returns else ""}"""
    def visit(body, parent, in_class=False):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                syms.append((n.name, "method" if in_class else "function", n.lineno, n.end_lineno, parent, sig(n), _first(ast.get_docstring(n))))
                for d in n.decorator_list:   # @router.get("/path"), @app.post("/x")
                    if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr in ("get", "post", "put", "delete", "patch", "websocket", "api_route") and d.args and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str):
                        syms.append((d.args[0].value, "route", n.lineno, n.end_lineno, n.name, f"""{d.func.attr.upper()} {d.args[0].value} -> {n.name}""", ""))
                visit(n.body, f"{parent}.{n.name}" if parent else n.name) if any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for x in n.body) else None
            elif isinstance(n, ast.ClassDef):
                syms.append((n.name, "class", n.lineno, n.end_lineno, parent, f"""class {n.name}({", ".join(ast.unparse(b) for b in n.bases)})""", _first(ast.get_docstring(n))))
                visit(n.body, f"{parent}.{n.name}" if parent else n.name, True)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)) and not parent:
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name) and re.fullmatch(r"_?[A-Z][A-Z0-9_]*", t.id): syms.append((t.id, "constant", n.lineno, n.end_lineno, "", ast.unparse(n)[:160], ""))
    visit(tree.body, "")
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call): refs.append((n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else "", n.lineno, "call"))
        elif id(n) in called: continue
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load): refs.append((n.id, n.lineno, "use"))
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load): refs.append((n.attr, n.lineno, "use"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            if _WORD.match(n.value): refs.append((n.value, n.lineno, "intent" if _INTENT_KEY.match(n.value) else "str"))   # "finance_rec_cell" in a dict, or "rec_cell" given to a helper that adds the prefix
            elif len(n.value) > 6: _strings(n.value, n.lineno, syms, refs)
        elif isinstance(n, ast.Dict):   # an intent table: {"finance_rec_cell": [_h_rec_cell], ...}
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and _INTENT_KEY.match(k.value) and isinstance(v, ast.List) and v.elts and isinstance(v.elts[0], ast.Name): syms.append((k.value, "intent", k.lineno, k.lineno, "", f"""intent {k.value} -> {", ".join(e.id for e in v.elts if isinstance(e, ast.Name))}""", ""))
        elif isinstance(n, ast.DictComp) and isinstance(n.key, ast.JoinedStr) and n.key.values and isinstance(n.key.values[0], ast.Constant):   # {f"agent_{k}": [fn] for k, fn in {"home": _h_home, ...}.items()}
            src = n.generators[0].iter
            if isinstance(src, ast.Call) and isinstance(src.func, ast.Attribute) and isinstance(src.func.value, ast.Dict):
                for k, v in zip(src.func.value.keys, src.func.value.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Name): syms.append((f"{n.key.values[0].value}{k.value}", "intent", k.lineno, k.lineno, "", f"""intent {n.key.values[0].value}{k.value} -> {v.id}""", ""))
    _sections(text.splitlines(), syms)
    return _first(ast.get_docstring(tree)), syms, [r for r in refs if r[0]]

def _js(text: str) -> tuple:
    syms, refs, lines = [], [], text.splitlines()
    for i, l in enumerate(lines, 1):
        for rx in _JS_DEFS:
            if (m := rx.match(l)): syms.append((m.group(1), "function", i, 0, "", l.strip()[:160], "")); break
        refs.extend((m.group(1), i, "call") for m in _JS_CALL.finditer(l) if m.group(1) not in _JS_SKIP)
    _sections(lines, syms)
    _strings(text, 1, syms, refs)
    return next((l.strip(" /*") for l in lines[:5] if l.strip().startswith(("//", "/*"))), ""), syms, refs

def _md(text: str) -> tuple:
    syms = [(m.group(2).strip(), "section", i, 0, "", "#" * len(m.group(1)), "") for i, l in enumerate(text.splitlines(), 1) if (m := _MD_HEAD.match(l))]
    return (syms[0][0] if syms else ""), syms, []

def _plain(text: str) -> tuple:
    syms, refs = [], []
    _sections(text.splitlines(), syms)
    _strings(text, 1, syms, refs)
    return "", syms, refs

def parse(rel: str, text: str) -> tuple:
    """(summary, symbols, refs, problem) for one file's text; a file that does not parse keeps what the patterns find and says why."""
    lang = _LANG.get(Path(rel).suffix.lower(), "")
    try: return (*{"py": _py, "js": _js, "md": _md}.get(lang, _plain)(text), "")
    except (SyntaxError, ValueError, RecursionError) as e: return (*_plain(text), f"{type(e).__name__}: {e}")

# --- Index ---

class CodeIndex:
    """One folder's map. skip(rel: Path) -> bool excludes files and folders (the agent's Files exclude)."""
    def __init__(self, root, skip=lambda rel: False):
        self.root, self.skip = Path(str(root)).resolve(), skip
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.db = INDEX_DIR / (re.sub(r"\W+", "_", str(self.root)).strip("_") + ".db")
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db)
        c.row_factory = sqlite3.Row
        c.executescript(_SCHEMA)
        return c
    def _walk(self):
        for d, dirs, files in os.walk(self.root):
            base = Path(d).relative_to(self.root)
            dirs[:] = sorted(x for x in dirs if not self.skip(base / x))
            for f in sorted(files):
                if Path(f).suffix.lower() in _LANG and not self.skip(base / f): yield (base / f).as_posix()
    def update(self) -> dict:
        """Brings the index up to date with the folder: new and changed files parsed, removed ones dropped. -> counts and seconds."""
        t0, c, parsed = time.time(), self._conn(), 0
        if (c.execute("SELECT v FROM meta WHERE k='version'").fetchone() or [None])[0] != _VERSION: c.execute("UPDATE files SET mtime=-1"); c.execute("REPLACE INTO meta VALUES ('version', ?)", (_VERSION,))
        known = {r["path"]: (r["mtime"], r["size"]) for r in c.execute("SELECT path, mtime, size FROM files")}
        seen = set()
        for rel in self._walk():
            seen.add(rel)
            st = (self.root / rel).stat()
            if known.get(rel) == (st.st_mtime, st.st_size): continue
            self._store(c, rel, st, (self.root / rel).read_text(errors="replace") if st.st_size <= _MAX_BYTES else "")
            parsed += 1
        for rel in known.keys() - seen: self._drop(c, rel)
        c.commit()
        c.close()
        return {"files": len(seen), "parsed": parsed, "removed": len(known.keys() - seen), "seconds": round(time.time() - t0, 2)}
    def _drop(self, c, rel: str):
        for t in ("files", "symbols", "refs", "search"): c.execute(f"DELETE FROM {t} WHERE path=?", (rel,))
    def _store(self, c, rel: str, st, text: str):
        summary, syms, refs, problem = parse(rel, text) if text else ("", [], [], "too large to parse")
        self._drop(c, rel)
        c.execute("INSERT INTO files VALUES (?,?,?,?,?,?,?)", (rel, st.st_mtime, st.st_size, text.count("\n") + 1 if text else 0, _LANG.get(Path(rel).suffix.lower(), ""), summary, problem))
        c.executemany("INSERT INTO symbols VALUES (?,?,?,?,?,?,?,?)", [(rel, *s) for s in syms])
        c.executemany("INSERT INTO refs VALUES (?,?,?,?)", [(rel, *r) for r in refs])
        c.executemany("INSERT INTO search VALUES (?,?,?,?,?)", [(rel, s[0], s[1], f"{_words(s[0])} {s[5]} {s[6]}", s[2]) for s in syms] + [(rel, Path(rel).name, "file", f"{_words(Path(rel).stem)} {summary}", 1)])
    # --- Queries (text for the agent) ---
    def find(self, name: str, kind: str = "") -> str:
        c, k = self._conn(), " AND kind=?" if kind else ""
        rows = c.execute(f"SELECT * FROM symbols WHERE name=?{k} ORDER BY path, line", (name, *([kind] if kind else []))).fetchall() or c.execute(f"SELECT * FROM symbols WHERE name LIKE ?{k} ORDER BY length(name), path LIMIT 40", (f"%{name}%", *([kind] if kind else []))).fetchall()
        c.close()
        return "\n".join(f"""{r["kind"]} {r["name"]}  {r["path"]}:{r["line"]}{"-" + str(r["end_line"]) if r["end_line"] and r["end_line"] != r["line"] else ""}{"  in " + r["parent"] if r["parent"] else ""}{"  " + r["signature"] if r["signature"] and r["kind"] not in ("section",) else ""}{"  - " + r["doc"] if r["doc"] else ""}""" for r in rows) or f"no definition named like {name}"
    def refs(self, name: str) -> str:
        c = self._conn()
        defs = c.execute("SELECT path, line, kind FROM symbols WHERE name=?", (name,)).fetchall()
        short = name.split("_", 1)[1] if any(d["kind"] == "intent" for d in defs) else ""   # an intent is often sent through a helper that adds its prefix: _vals("rec_cell") for finance_rec_cell
        rows = c.execute("SELECT path, kind, group_concat(line, ', ') AS lines FROM (SELECT path, line, CASE WHEN name=? THEN kind ELSE 'short name ' || name END AS kind FROM refs WHERE name=? OR (name=? AND kind='str') ORDER BY path, line) GROUP BY path, kind ORDER BY path", (name, name, short)).fetchall()
        c.close()
        head = "defined: " + ", ".join(f"""{d["path"]}:{d["line"]} ({d["kind"]})""" for d in defs) if defs else "defined: (not in this folder - a library, a builtin, or a misspelling)"
        return "\n".join([head, *(f"""{r["path"]}: {r["kind"]} at lines {r["lines"]}""" for r in rows[:150])] + ([f"[... {len(rows) - 150} more files]"] if len(rows) > 150 else [])) if rows or defs else f"{name}: not defined or used anywhere in this folder"
    def file_map(self, rel: str) -> str:
        c = self._conn()
        f = c.execute("SELECT * FROM files WHERE path=?", (rel,)).fetchone()
        rows = c.execute("SELECT * FROM symbols WHERE path=? AND kind NOT IN ('css') ORDER BY line", (rel,)).fetchall()
        css = c.execute("SELECT count(*) FROM symbols WHERE path=? AND kind='css'", (rel,)).fetchone()[0]
        c.close()
        if not f: return f"{rel}: not in the index (not a code or text file, excluded, or new since the last update)"
        head = f"""{rel} - {f["lines"]} lines{" - " + f["summary"] if f["summary"] else ""}{" - PROBLEM: " + f["problem"] if f["problem"] else ""}{f" - {css} CSS classes defined" if css else ""}"""
        seen, out = set(), [head]
        for r in rows:
            if (r["line"], r["signature"]) in seen and r["signature"]: continue   # A = B = None: one line, not one per name
            seen.add((r["line"], r["signature"]))
            out.append(f"""{r["line"]}{"-" + str(r["end_line"]) if r["end_line"] and r["end_line"] != r["line"] else ""}: {"  " * (r["parent"].count(".") + 1) if r["parent"] else ""}{"## " + r["name"] if r["kind"] == "section" else r["signature"] or r["name"]}{"  - " + r["doc"] if r["doc"] else ""}""")
        return "\n".join(out)
    def search(self, query: str, limit: int = 25) -> str:
        words = re.findall(r"\w+", query.replace("_", " "))
        if not words: return "give some words to search for"
        c = self._conn()
        def run(op): return c.execute("SELECT path, name, kind, line, snippet(search, 3, '[', ']', '...', 12) AS snip FROM search WHERE search MATCH ? ORDER BY bm25(search, 1.0, 4.0, 1.0, 2.0) LIMIT ?", (f" {op} ".join(f'"{w}"' for w in words), limit)).fetchall()
        rows = run("AND") or run("OR")
        c.close()
        return "\n".join(f"""{r["path"]}:{r["line"]}  {r["kind"]} {r["name"]}  {r["snip"]}""" for r in rows) or "(nothing found)"
    def span(self, name: str, rel: str = "") -> list:
        """[(path, line, end_line, kind)] of definitions named name (in rel when given) - read_symbol shows their source."""
        c = self._conn()
        rows = [tuple(r) for r in c.execute(f"""SELECT path, line, end_line, kind FROM symbols WHERE name=? AND kind IN ('function','method','class','constant'){" AND path=?" if rel else ""} ORDER BY path, line""", (name, *([rel] if rel else [])))]
        c.close()
        return rows
