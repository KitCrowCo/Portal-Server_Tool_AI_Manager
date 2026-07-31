""" steps.py — Step type registry + built-in step implementations for ai_manager.
"Any AI operation is a step" - chat completion, a knowledge query, an insert, image generation, future TTS/STT, are all the same shape: async fn(config: dict, ctx: StepContext) -> dict.
The engine never special-cases behavior; it only sequences and reports.
Config values may reference upstream results with {node_id.key} templating, resolved by ctx.resolve() before the step runs - keeps step implementations free of graph-walking logic."""

import httpx, re, json, sys, asyncio, uuid
from pathlib import Path
from tools.ai_manager import engine

# To be generalized:
from tools.ai_manager.connections import get_conn, lightrag_query, lightrag_insert_text, _base, stream_llm, lightrag_list_entities, flux2_encode, flux2_generate, list_conns, list_models_sync

_STEP_TYPES: dict = {}
ENV: dict = {}
def init(env: dict):
    global ENV
    ENV = env

def register_step_type(name, fn, label="", config_schema=None, output_keys=None, guide=""): _STEP_TYPES[name] = {"fn": fn, "label": label or name, "config_schema": config_schema or [], "output_keys": output_keys or [], "guide": guide} # label/config_schema are optional UI hints for pipeline builders - the engine itself never reads them.
def get_step_type(name: str) -> dict: return _STEP_TYPES.get(name)
def list_step_types() -> list: return [{"type": k, **{kk: vv for kk, vv in v.items() if kk != "fn"}} for k, v in _STEP_TYPES.items()]
def _llm_conn_options(values=None): return [("", "(none)")] + [(c["_id"], f'{c.get("display_name",c["_id"])} [{c.get("connection_type")}]') for c in (list_conns("ollama") + list_conns("vllm"))]
def _knowledge_conn_options(values=None): return [("", "(none)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in list_conns("lightrag")]
def _flux2_text_options(values=None): return [("", "(none)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in list_conns("flux2_text")]
def _flux2_image_options(values=None): return [("", "(none)")] + [(c["_id"], c.get("display_name",c["_id"])) for c in list_conns("flux2_image")]
def _pipeline_options(values=None): return [("", "(none)")] + [(p["id"], f'{p.get("name",p["id"])} [{", ".join(p.get("tags",[]))}]') for p in engine.list_pipelines()]

def _model_options_for_config(values=None):
    """Reads the config's OWN conn_id at render time to list that connection's real models."""
    conn_id = (values or {}).get("conn_id", "")
    if not conn_id: return []
    conn = get_conn(conn_id)
    models = list_models_sync(conn) if conn else []
    cur = (values or {}).get("model", "")
    if cur and cur not in models: models = [cur] + models
    return [(m, m) for m in models]

def register_builtins():
    BI = ENV["tools"]["built_ins"]
    def _rmap_field(): return BI.SettingField("result_map", "Result Mapping (JSON, optional)", type="json", default={}, advanced=True, hint='Map this step\'s output fields to scratch keys, e.g. {"text":"article_md"}. "*" merges everything flat. {"text":{"key":"notes","mode":"append"}} accumulates rather than overwrites.')

    register_step_type("llm_generate", step_llm_generate, "LLM Generate (chat / decision / router)", [
        BI.SettingField("conn_id", "Connection", type="select", options=_llm_conn_options),
        BI.SettingField("model", "Model", type="select", options=_model_options_for_config),
        BI.SettingField("system_prompt", "System Prompt", type="textarea", default="You are a helpful AI assistant."),
        BI.SettingField("user_template", "User Prompt Template", type="textarea", default="{input}", hint="Reference an upstream node with {alias.field}, e.g. {n1.text}. Bare {input} is this pipeline's own input."),
        BI.SettingField("temperature", "Temperature", type="number", default=0.7, hint="No artificial step limit - test whatever value your task needs."),
        BI.SettingField("num_ctx", "Context Window (tokens)", type="number", default=16384, step=1),
        BI.SettingField("num_predict", "Max Output Tokens", type="number", default=-1, step=1, hint="-1 = unlimited (provider default)"),
        BI.SettingField("think", "Enable Thinking Mode", type="checkbox", default=False, hint="Only takes effect on connections whose profile declares supports_thinking (Ollama does). Silently ignored otherwise."),
        BI.SettingField("json_fields", "JSON Output Fields (comma-sep)", type="text", advanced=True, hint='e.g. "char_name, char_desc" - asks the model for exactly these named fields in one pass instead of one text blob. Falls back to plain {alias.text} if parsing fails.'),
        BI.SettingField("enforce_options", "Enforced Options (comma-sep)", type="text", advanced=True, hint="Constrains the reply to one of these words - enables Routing JSON below."),
        BI.SettingField("routes_json", "Routing JSON", type="json", default={}, advanced=True, hint='Maps an enforced-option word to a downstream node id, e.g. {"yes":"n5","no":"n8"}. Requires Enforced Options.'),
        BI.SettingField("top_p", "Top P", type="number", default=None, advanced=True),
        BI.SettingField("top_k", "Top K", type="number", default=None, step=1, advanced=True),
        _rmap_field()], output_keys=["text", "thinking", "choice"],
        guide="""Calls a chat model. With no config beyond Connection/Model, replies to {input} with the system prompt shown - this is the safe, always-works default.

Returns {alias.text} (full reply) and {alias.thinking} (if the model supports it). alias = this node's Reference Name field.

JSON Output Fields (Advanced): ask for several named values in one pass instead of one blob - e.g. "char_name, char_desc" gives {alias.char_name} and {alias.char_desc} directly.

Enforced Options + Routing JSON (Advanced): constrains the reply to one word from a list and routes to a different downstream node per word - for yes/no branches or category routers. Nothing else on this node needs Enforced Options to work; it's purely additive.""")

    register_step_type("find", step_find, "Find (grep, regex search)", [
        BI.SettingField("pattern","Regex Pattern","text", hint="Required - no default match makes sense for a search step."),
        BI.SettingField("source_key","Source Scratch Key","text",default="input"),
        BI.SettingField("path","File Path Override","text",advanced=True, hint="If set, reads from this file instead of Source Scratch Key."),
        BI.SettingField("dotall","Regex DOTALL","checkbox",default=False,advanced=True),
        _rmap_field()], output_keys=["matches", "count", "source"],
        guide="""Regex search, no LLM call. Returns {alias.matches} (list of {text, groups, start, end}), {alias.count}, and {alias.source} (the full searched text - useful as input to a downstream text_replace).

Default source is {input}; point Source Scratch Key at another node's output (e.g. "n2") to search that instead.""")

    register_step_type("text_replace", step_text_replace, "Text Replace (sed / template fill)", [
        BI.SettingField("mode","Mode","select",default="replace_all", options=[("replace_all","Replace all matches"),("replace_first","Replace first match only")]),
        BI.SettingField("pattern","Regex Pattern","text"),
        BI.SettingField("replacement_template","Replacement Template","text", hint="{match} = whole match, {0}/{1}... = capture groups."),
        BI.SettingField("source_key","Source Scratch Key","text",default="input"),
        BI.SettingField("matches_key","Matches Key (sequence mode)","text",advanced=True, hint="Set this + Replacements Key to substitute a prior Find step's matches in order instead of using Pattern."),
        BI.SettingField("replacements_key","Replacements Key (sequence mode)","text",advanced=True),
        _rmap_field()],
        guide="""Two modes. Pattern mode (default): regex + replacement template against Source Scratch Key. Sequence mode (Advanced): point Matches Key at a prior Find step's {alias.matches} and Replacements Key at a list of replacement strings - substitutes each match in document order, e.g. one generated image path per flagged placeholder.

Returns {alias.text}.""")

    register_step_type("knowledge_query", step_knowledge_query, "Knowledge Query", [
        BI.SettingField("conn_id", "Knowledge Connection", type="select", options=_knowledge_conn_options),
        BI.SettingField("query_template","Query Template","textarea",default="{input}"),
        BI.SettingField("mode","Search Mode","select",default="hybrid", options=[("hybrid","Hybrid (default, recommended)"),("local","Local"),("global","Global"),("naive","Naive"),("mix","Mix")]),
        BI.SettingField("return_context_only","Return raw retrieved context only (no re-synthesis)","checkbox",default=False,advanced=True),
        BI.SettingField("extra_params","Extra Params (JSON passthrough)","json",default={},advanced=True, hint="Anything this dashboard doesn't expose yet (top_k, chunk_top_k, etc.) - passed straight to the LightRAG query."),
        _rmap_field()], output_keys=["response", "raw"],
        guide="""Queries a LightRAG knowledge base with hybrid mode by default. Returns {alias.response} (synthesized answer) and {alias.raw} (full API response, for anything not surfaced directly).""")

    register_step_type("knowledge_list_entities", step_knowledge_list_entities, "Knowledge Graph - List Entities", [
        BI.SettingField("conn_id","Knowledge Connection",type="select", options=_knowledge_conn_options),
        BI.SettingField("limit","Limit","number",default=500,step=1,advanced=True),
        _rmap_field()], output_keys=["entities"],
        guide="""Reads the knowledge graph's own entity list directly - no LLM call, no re-deriving what LightRAG already indexed. Returns {alias.entities}, typically fed straight into a foreach_call_pipeline.""")

    register_step_type("knowledge_insert_text", step_knowledge_insert_text, "Knowledge Insert Text", [
        BI.SettingField("conn_id","Knowledge Connection", type="select", options=_knowledge_conn_options),
        BI.SettingField("text_template","Text Template","textarea",default="{input}"),
        BI.SettingField("source_label","Source Label","text",advanced=True)],
        guide="""Writes text directly into the knowledge base (not shadow-staged - LightRAG has no diff/review concept of its own; treat this node as a real, immediate write).""")

    register_step_type("file_write", step_file_write, "File Write (shadow-staged)", [
        BI.SettingField("path","Destination Path","text", hint="Supports {templates}, e.g. articles/{input}.md."),
        BI.SettingField("content_key","Content Scratch Key","text",default="text", hint='Either a Result-Mapped name, or a direct {n2.text} reference. Bare "text" only works if exactly one upstream node was mapped to that name.'),
        BI.SettingField("fm_root","FS Root","file_picker",default="./data/_common",advanced=True),
        _rmap_field()],
        guide="""Never writes directly - always through the Shadow Stage (Tessa's Pending Reviews / bottom-bar Shadow Diff), where you accept or reject before it touches the real file. Errors loudly (rather than writing an empty file) if Location or Content Scratch Key can't resolve.""")

    register_step_type("file_write_binary", step_file_write_binary, "File Write Binary (shadow-staged)", [
        BI.SettingField("path","Destination Path","text"),
        BI.SettingField("content_key","Content Scratch Key","text",default="image_file"),
        BI.SettingField("fm_root","FS Root","file_picker",default="./data/_common",advanced=True),
        BI.SettingField("source_root","Source Root","file_picker",default=".",advanced=True),
        _rmap_field()],
        guide="""Same shadow-staged write as File Write, for binary output (typically an image_generate node's result). Content Scratch Key should point at a node whose output includes a file_name field.""")

    register_step_type("python_exec", step_python_exec, "Python Script", [
        BI.SettingField("script_body","Inline Script Body","textarea", hint="Quick glue logic. Receives one argument (Input Template, resolved), must print a JSON object as its last stdout line."),
        BI.SettingField("input_template","Input Template","textarea",default="{input}"),
        BI.SettingField("script_path","Script File (instead of inline)","file_picker",advanced=True),
        BI.SettingField("timeout_s","Timeout (s)","number",default=600,step=1,advanced=True, hint="Generous by default for CPU-bound/no-GPU hardware - raise further if needed, no artificial ceiling."),
        _rmap_field()],
        guide="""Runs either Inline Script Body or Script File as a subprocess with Input Template as argv[1]. Whatever the script prints as its final JSON line becomes this node's result dict. No default script - this node always needs at least Inline Script Body or Script File filled in.""")

    register_step_type("list_files", step_list_files, "List Files", [
        BI.SettingField("root","Target Directory","text",default="./data/ai_tools/_knowledge"),
        BI.SettingField("extensions","Extensions (comma-sep)","text",advanced=True, hint="Empty = all files."),
        _rmap_field()], output_keys=["files"],
        guide="""Lists relative file paths under a directory. No LLM call. Default root is the shared knowledge folder. Returns {alias.files}, typically feeding foreach_call_pipeline or chunked_file_pass.""")

    register_step_type("echo", step_echo, "Echo / Passthrough", [
        BI.SettingField("template","Template","textarea",default="{input}"),
        _rmap_field()],
        guide="""Resolves a template and passes it through unchanged - default is a no-op pass of {input}. Useful for renaming/reshaping a value between two nodes without an LLM call, or as a merge-point placeholder.""")

    register_step_type("expr", step_expr, "Expression (small derived value)", [
        BI.SettingField("expr","Python Expression","text",default="input", hint="Restricted builtins only (len/str/int/float/min/max/sorted/round) - no import, no file/network access."),
        BI.SettingField("vars_json","Variables (JSON: name -> template)","json",default={},advanced=True, hint='e.g. {"a":"{n1.text}"} makes "a" available in the expression.'),
        _rmap_field()],
        guide="""One Python expression, safely sandboxed - for glue too small to justify python_exec (deriving a slug, basic arithmetic on a numeric result). Default expr="input" just passes {input} through as a sanity-check default.""")

    register_step_type("call_pipeline", step_call_pipeline, "Call Another Pipeline", [
        BI.SettingField("pipeline_id", "Pipeline ID", type="select", options=_pipeline_options),
        BI.SettingField("input_template", "Input Template", type="textarea", default="{input}"),
        BI.SettingField("extra_inputs_json", "Extra Inputs (JSON: key -> template)", type="json", default={}, advanced=True, hint='Passed alongside "input" into the sub-pipeline, e.g. {"source_dir":"{n1.files}"}.'),
        BI.SettingField("_call_depth", "Call Depth Override", type="number", default=0, advanced=True, hint="Advanced/internal - leave at 0 unless debugging recursion depth."),
        _rmap_field()],
        guide="""Runs a saved pipeline to completion inline and folds its final scratch back as this node's result. No default pipeline - Pipeline ID must be set.""")

    register_step_type("foreach_call_pipeline", step_foreach_call_pipeline, "For Each Item, Call Pipeline", [
        BI.SettingField("items_source", "Items Scratch Key", type="text", hint="Scratch key containing a JSON array or newline-separated text block."),
        BI.SettingField("pipeline_id", "Pipeline ID", type="select", options=_pipeline_options),
        BI.SettingField("extra_inputs_json", "Extra Inputs (JSON: key -> template)", type="json", default={}, advanced=True),
        BI.SettingField("_call_depth", "Call Depth Override", type="number", default=0, advanced=True),
        _rmap_field()], output_keys=["results","count"],
        guide="""Items Scratch Key is a LITERAL field name, not a {template} - no braces, no dots, no alias/node id.

By default every step's output fields flatten onto top-level scratch under their own name. A list_files node's output {"files": [...]} becomes available at the plain key "files" with zero config needed.

WRONG (common mistake): typing the upstream node's alias or id (e.g. "docs" or "n_1a2b3c4d") - that key holds the WHOLE result dict, not the list inside it, and this step will silently run zero iterations.
RIGHT: type the flattened field name itself - "files", "entities", "results", whatever the upstream step's output_keys list shows.

Worked example:
  Node A (list_files, root=./data/ai_tools/_knowledge) -> flattens to scratch key "files" (a list of path strings)
  Node B (this step): Items Scratch Key = files | Pipeline ID = <a saved sub-pipeline>
  -> runs the sub-pipeline once per path string in that list, returns {alias.results} (list of one output per item) and {alias.count}

If two upstream nodes both produce a "files" key you'll get a silent collision (last one written wins) - set that node's Result Mapping to {"files":"unique_name"} to rename it before using it here.""")

    register_step_type("branch_on", step_branch_on, "Branch On Decision", [
        BI.SettingField("decision_key", "Decision Scratch Key", type="text", default="decision"),
        BI.SettingField("routes_json", "Routes Map (JSON)", type="json", default={}, hint='e.g. {"yes":"pl_abc","no":"pl_def"} - maps a decision string to a pipeline id.'),
        BI.SettingField("default_pipeline_id", "Default Pipeline ID", type="text", advanced=True, hint="Fallback if the decision value doesn't match any route."),
        BI.SettingField("input_template", "Input Template", type="textarea", default="{input}", advanced=True),
        BI.SettingField("extra_inputs_json", "Extra Inputs (JSON)", type="json", default={}, advanced=True),
        _rmap_field()],
        guide="""Reads Decision Scratch Key (typically an upstream llm_generate's {alias.choice}) and runs whichever sub-pipeline Routes Map assigns to that value. Errors loudly if the value matches nothing and no Default Pipeline ID is set.""")

    register_step_type("image_generate", step_image_generate, "Text-to-Image (Flux2)", [
        BI.SettingField("text_encoder_conn_id", "Text Encoder Connection", type="select", options = _flux2_text_options),
        BI.SettingField("image_conn_id", "Image Connection", type="select", options = _flux2_image_options),
        BI.SettingField("prompt_template", "Prompt Template", type="textarea", default="{input}"),
        BI.SettingField("width", "Width", type="number", default=1024, step=16),
        BI.SettingField("height", "Height", type="number", default=1024, step=16),
        BI.SettingField("steps", "Steps", type="number", default=4, step=1, hint="4 for distilled/fast models, 20+ for base models."),
        BI.SettingField("cfg", "Guidance Scale (CFG)", type="number", default=1.0, step="any", advanced=True),
        BI.SettingField("shift", "Shift", type="number", default=1.0, step="any", advanced=True),
        BI.SettingField("seed", "Seed", type="number", default=-1, advanced=True, hint="-1 for random."),
        _rmap_field()], output_keys=["file_name"],
        guide="""Two-stage: encodes the prompt on the text-encoder connection, then generates on the image connection. Returns {alias.file_name}, typically fed to file_write_binary.""")

    register_step_type("edit_in_place", step_edit_in_place, "Edit In Place (gap-aware, chunked)", [
        BI.SettingField("conn_id", "Connection", type="select", options=_llm_conn_options),
        BI.SettingField("model", "Model", type="select", options=_model_options_for_config),
        BI.SettingField("source_key", "Source Scratch Key", type="text", default="input"),
        BI.SettingField("gap_marker", "Gap Marker", type="text", default="[[GAP]]", hint="Text before this marker gets chunk-rewritten; text after is used as continuation context."),
        BI.SettingField("system_prompt", "System Prompt", type="textarea", advanced=True),
        BI.SettingField("rewrite_prompt", "Rewrite Prompt", type="textarea", advanced=True, hint="Default: light grammar/flow/continuity pass, same length and content."),
        BI.SettingField("continuation_prompt", "Continuation Prompt", type="textarea", advanced=True),
        BI.SettingField("open_ended_prompt", "Open Ended Prompt", type="textarea", advanced=True),
        BI.SettingField("chunk_tokens", "Chunk Tokens", type="number", default=4000, step=1, advanced=True),
        BI.SettingField("num_ctx", "Context Window", type="number", default=8192, step=1, advanced=True),
        BI.SettingField("temperature", "Temperature", type="number", default=0.3, step="any", advanced=True),
        _rmap_field()], output_keys=["text","gap_remaining"],
        guide="""Chunk-rewrites Source Scratch Key in place. With no Gap Marker present, does a straight chunked rewrite pass. With a marker, rewrites everything before it and bridges to whatever's after it (or continues open-ended if nothing follows). Sensible defaults on all prompts - only Connection/Model are required.""")

    register_step_type("chunked_file_pass", chunked_file_pass, "Chunked File Pass", [
        BI.SettingField("conn_id", "Connection", type="select", options=_llm_conn_options),
        BI.SettingField("model", "Model", type="select", options=_model_options_for_config),
        BI.SettingField("items_source", "Files Scratch Key", type="text", default="files", hint="Typically a list_files node's {alias.files}."),
        BI.SettingField("user_template", "User Template", type="textarea", default="{chunk_content}", hint="Also available: {file_name}, {file_path}, {chunk_number}, {chunks_total}."),
        BI.SettingField("system_prompt", "System Prompt", type="textarea", advanced=True),
        BI.SettingField("chunk_tokens", "Chunk Tokens", type="number", default=6000, step=1, advanced=True),
        BI.SettingField("model_ctx", "Context Window", type="number", default=32768, step=1, advanced=True),
        BI.SettingField("temperature", "Temperature", type="number", default=0.3, step="any", advanced=True),
        BI.SettingField("output_separator", "Output Separator", type="text", default="\n\n---\n\n", advanced=True),
        BI.SettingField("fm_root", "FS Root", type="file_picker", default="./data/_common", advanced=True),
        _rmap_field()], output_keys=["text","files_processed"],
        guide="""Runs one LLM pass per chunk, per file, over a list of files (e.g. from list_files). Concatenates all outputs into {alias.text}. No default prompt beyond passing the chunk through - set User Template for real use.""")

    register_step_type("chunked_synthesis", chunked_synthesis, "Chunked Synthesis", [
        BI.SettingField("conn_id", "Connection", type="select", options=_llm_conn_options),
        BI.SettingField("model", "Model", type="select", options=_model_options_for_config),
        BI.SettingField("source_key", "Source Scratch Key", type="text", default="input"),
        BI.SettingField("user_template", "User Template", type="textarea", default="{chunk_content}"),
        BI.SettingField("system_prompt", "System Prompt", type="textarea", advanced=True),
        BI.SettingField("chunk_tokens", "Chunk Tokens", type="number", default=6000, step=1, advanced=True),
        BI.SettingField("model_ctx", "Context Window", type="number", default=32768, step=1, advanced=True),
        BI.SettingField("temperature", "Temperature", type="number", default=0.3, step="any", advanced=True),
        BI.SettingField("output_separator", "Output Separator", type="text", default="\n\n---\n\n", advanced=True),
        _rmap_field()], output_keys=["text","chunks_processed"],
        guide="""Same chunked LLM-pass pattern as Chunked File Pass, but over one long value (Source Scratch Key) instead of a list of files - for summarizing/transforming one large document.""")

    register_step_type("format_each", step_format_each, "Format Each (list -> repeated text)", [
        BI.SettingField("items_key","Items Scratch Key","text", hint="Typically foreach_call_pipeline's {alias.results}."),
        BI.SettingField("item_template","Item Template","textarea",default="{input}", hint="Uses {field} for keys INSIDE one list item directly - not {node.field}, since a list item isn't a node."),
        BI.SettingField("separator","Separator","text",default="\n\n---\n\n",advanced=True),
        _rmap_field()], output_keys=["text","count"],
        guide="""Turns a list of dicts into one repeated-template text block. No LLM call. Feeds nicely into a file_write after a foreach_call_pipeline fan-out.""")

async def step_llm_generate(config: dict, ctx) -> dict:
    """Universal text-generation node. Returns {"text": full reply, "choice": <if enforce_options set>}.
    enforce_options + routes_json additionally yields _chosen_next for in-graph routing.
    All wire format lives in connections.stream_llm() via the connection's own profile - agnostic to provider."""
    conn = get_conn(config.get("conn_id", ""))
    if not conn: raise RuntimeError("llm_generate: no connection configured")
    model = config.get("model", "")
    if not model: raise RuntimeError("llm_generate: no model configured")
    raw_opts = config.get("enforce_options", "")
    options = raw_opts if isinstance(raw_opts, list) else [o.strip() for o in raw_opts.split(",") if o.strip()]
    json_fields = [f.strip() for f in str(config.get("json_fields","")).split(",") if f.strip()]
    sys_p = ctx.resolve(config.get("system_prompt") or "")
    if options: sys_p = (sys_p + f"\n\nRespond with exactly one of these words and nothing else: {', '.join(options)}").strip()
    if json_fields: sys_p = (sys_p + f"\n\nRespond ONLY with a single JSON object with exactly these keys: {json.dumps(json_fields)}. No markdown fences, no text before or after the JSON.").strip()
    messages = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content": ctx.resolve(config.get("user_template") or "{input}")}]
    full = ""
    thinking_full = ""
    async for text, thinking in stream_llm(conn, messages, model, think=config.get("think", False), temperature=config.get("temperature", 0.3), num_ctx=config.get("num_ctx", 8192), num_predict=config.get("num_predict", -1), top_p=config.get("top_p"), top_k=config.get("top_k")):
        full += text; thinking_full += thinking
        await ctx.stream("text", text)
    result = {"text": full, "thinking": thinking_full}
    if json_fields:
        parsed = _extract_json_fields(full, json_fields)
        if parsed: result.update(parsed)
        else: await ctx.progress("json_fields requested but parsing failed - raw text kept under 'text'")
    if not options: return result
    choice = next((o for o in options if o.lower() in full.lower()), options[0])
    result["choice"] = choice
    try: routes = json.loads(config.get("routes_json","{}") or "{}")
    except Exception: routes = {}
    if routes: result["_chosen_next"] = routes.get(choice)
    return result

def _extract_json_fields(text: str, fields: list) -> dict | None:
    """Best-effort JSON extraction from an LLM reply - strips a wrapping markdown fence if present, finds the first {...} block, returns only the requested keys that were actually present. Returns None (not {}) on total failure so the caller can fall back to plain text."""
    cleaned = re.sub(r'\A```(?:json)?\s*|\s*```\Z', '', text.strip())
    m = re.search(r'\{.*\}', cleaned, re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    found = {k: obj[k] for k in fields if k in obj}
    return found or None

async def step_find(config: dict, ctx) -> dict:
    """Returns {"matches": [...], "count": n, "source": full searched text}."""
    pattern = config.get("pattern", "")
    if not pattern: raise RuntimeError("find: pattern required")
    content = Path(ctx.resolve(config["path"])).read_text(encoding="utf-8", errors="ignore") if config.get("path") else str(ctx.scratch.get(config.get("source_key","input"), ""))
    flags = re.DOTALL if config.get("dotall") else 0
    matches = [{"text": m.group(0), "groups": list(m.groups()), "start": m.start(), "end": m.end()} for m in re.finditer(pattern, content, flags)]
    return {"matches": matches, "count": len(matches), "source": content}

async def step_text_replace(config: dict, ctx) -> dict:
    """Returns {"text": edited result}. Pattern mode: regex + replacement_template ({match}/{0}/{1}...). Sequence mode (matches_key + replacements_key): substitutes a prior find step's matches in order."""
    source = str(ctx.scratch.get(config.get("source_key","input"), ""))
    if config.get("matches_key"):
        matches, repls = ctx.scratch.get(config["matches_key"], []), ctx.scratch.get(config.get("replacements_key",""), [])
        if len(repls) != len(matches): raise RuntimeError(f"text_replace: {len(matches)} matches but {len(repls)} replacements")
        out, cursor = [], 0
        for m, r in zip(matches, repls): out.append(source[cursor:m["start"]]); out.append(str(r)); cursor = m["end"]
        out.append(source[cursor:])
        edited = "".join(out)
    else:
        def _sub(m):
            r = config.get("replacement_template","").replace("{match}", m.group(0))
            for i, g in enumerate(m.groups() or []): r = r.replace(f"{{{i}}}", g or "")
            return r
        edited = re.sub(config.get("pattern",""), _sub, source, count=(0 if config.get("mode","replace_all")=="replace_all" else 1))
    return {"text": edited}

async def step_list_files(config: dict, ctx) -> dict:
    """Lists files under a root as a JSON array of relative-path strings - typically feeds foreach_call_pipeline's items_source.
    Deterministic, no AI call - this is the kind of plain Python step the architecture is meant to mix freely with generation steps."""
    exts = tuple(e.strip().lower() for e in (config.get("extensions","") or "").split(",") if e.strip())
    root = Path(config.get("root", "./data/ai_tools/_knowledge"))
    return {"files": sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.is_file() and (not exts or f.suffix.lower() in exts))}

async def step_knowledge_query(config: dict, ctx) -> dict:
    """Returns {"response": synthesized answer text, "raw": full raw API response dict}.
    Set return_context_only=true to get LightRAG's retrieved-context-only mode (no LLM synthesis pass on LightRAG's own side)
    - useful when you want raw retrieved chunks/entities for your own downstream processing rather than another summarized answer. 
    extra_params (JSON) passes through untouched to lightrag_query for anything this dashboard doesn't special-case (top_k, chunk_top_k, whatever a given LightRAG version supports)."""
    conn = get_conn(config.get("conn_id", ""), conn_type="lightrag")
    if not conn: raise RuntimeError("knowledge_query: no knowledge connection configured")
    q = ctx.resolve(config.get("query_template", "{input}"))
    try: extra = json.loads(config.get("extra_params","{}") or "{}")
    except Exception: extra = {}
    if config.get("return_context_only"): extra["only_need_context"] = True
    r = await lightrag_query(conn, q, config.get("mode", "hybrid"), **extra)
    return {"response": r.get("response", r.get("error", "")), "raw": r}

async def step_knowledge_list_entities(config: dict, ctx) -> dict:
    """Returns the knowledge graph's own entity/node labels directly - no LLM call, no raw-file chunking.
    LightRAG has already built this list during ingestion; re-deriving 'what are the concepts in this corpus' via a chunked_synthesis pass over the raw files duplicates work the graph already did.
    Use this to feed foreach_call_pipeline directly. limit caps how many labels come back (LightRAG's graph listing endpoints vary by version - see connections.lightrag_graph_dot for the same underlying fetch)."""
    conn = get_conn(config.get("conn_id", ""), conn_type="lightrag")
    if not conn: raise RuntimeError("knowledge_list_entities: no knowledge connection configured")
    return {"entities": await lightrag_list_entities(conn, limit=config.get("limit", 500))}

async def step_image_generate(config: dict, ctx) -> dict:
    enc_conn, img_conn = get_conn(config.get("text_encoder_conn_id",""), conn_type="flux2_text"), get_conn(config.get("image_conn_id",""), conn_type="flux2_image")
    if not enc_conn or not img_conn: raise RuntimeError("image_generate: text encoder and image connections both required")
    prompt = ctx.resolve(config.get("prompt_template") or "{input}")
    job_id = f"{ctx.job_id}_{uuid.uuid4().hex[:6]}"
    enc = await flux2_encode(enc_conn, prompt, job_id=job_id, max_sequence_length=config.get("max_sequence_length", 512))
    if enc.get("error"): raise RuntimeError(f"image_generate encode: {enc['error']}")
    gen = await flux2_generate(img_conn, {"prompt": prompt, "embed_job_id": job_id, "width": config.get("width",1024), "height": config.get("height",1024), "steps": config.get("steps",4), "guidance_scale": config.get("cfg",1.0), "shift": config.get("shift",1.0), "seed": config.get("seed",-1), "model_path": config.get("model_path",""), "vae_path": config.get("vae_path","")})
    if gen.get("error"): raise RuntimeError(f"image_generate: {gen['error']}")
    return {"file_name": gen["file_name"]}

async def step_expr(config: dict, ctx) -> dict:
    """Mini derived-value node - a single Python expression evaluated against resolved scratch values, for glue logic too small to justify a whole python_exec script (deriving a slug/title from a long string, combining two upstream fields, basic arithmetic on a numeric result).
    Restricted builtins only (no import, no file/network access, no exec) - genuinely small expressions, not a general scripting sandbox; use python_exec for anything that needs real logic or external access.
    vars_json maps {local_name: "{node_id.field}" template} - each gets ctx.resolve()'d before eval."""
    try: var_templates = json.loads(config.get("vars_json","{}") or "{}")
    except Exception: var_templates = {}
    local_vars = {name: ctx.resolve(tpl) for name, tpl in var_templates.items()}
    safe_builtins = {"len": len, "str": str, "int": int, "float": float, "min": min, "max": max, "sorted": sorted, "round": round}
    try: value = eval(config.get("expr", "input"), {"__builtins__": safe_builtins}, {**local_vars, "input": ctx.resolve("{input}")})
    except Exception as e: raise RuntimeError(f"expr: {e}")
    return {"value": value}

def _chunk_text(text: str, chunk_tokens: int = 4000) -> list[str]:
    """Splits text into chunks roughly matching chunk_tokens (assuming ~4 chars per token)."""
    if not text: return []
    char_limit = chunk_tokens * 4
    paragraphs = text.split("\n\n")
    chunks, current_chunk, current_len = [], [], 0
    for p in paragraphs:
        if current_len + len(p) > char_limit and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p) + 2
    if current_chunk: chunks.append("\n\n".join(current_chunk))
    return chunks

async def step_edit_in_place(config: dict, ctx) -> dict:
    """In-place chunk rewrite with gap-fill continuation - operates purely on a scratch text key (source_key), not any specific document store.
    Any tool wanting to edit 'its own document' wraps this: load content into scratch under source_key, run this step, read result['text'] back out.
    Prompts are fully configurable - rewrite_prompt/continuation_prompt/open_ended_prompt, not hardcoded strings, so this generalizes past prose (a code-aware caller can supply its own instructions)."""
    content = str(ctx.scratch.get(config.get("source_key", "input"), ""))
    marker = config.get("gap_marker", "[[GAP]]")
    conn = get_conn(config.get("conn_id", ""))
    model = config.get("model", "")
    if not conn or not model: raise RuntimeError("edit_in_place: connection/model required")
    chunk_tokens = int(config.get("chunk_tokens", 4000))
    sys_p = ctx.resolve(config.get("system_prompt") or "")
    rewrite_tpl = config.get("rewrite_prompt") or "Rewrite this passage in place - same length and content, fix grammar/flow/continuity only:\n\n{chunk}"
    before, _, after = content.partition(marker) if marker in content else (content, "", "")
    chunks = _chunk_text(before, chunk_tokens) if before else []
    edited = []
    for i, chunk in enumerate(chunks):
        await ctx.progress(f"editing chunk {i+1}/{len(chunks)}")
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content": rewrite_tpl.replace("{chunk}", chunk)}]
        full = ""
        async for piece, _think in stream_llm(conn, msgs, model, temperature=config.get("temperature",0.3), num_ctx=config.get("num_ctx",8192), num_predict=config.get("num_ctx",8192)): full += piece
        edited.append(full.strip() or chunk)
    if marker in content and after.strip():
        fill_tpl = config.get("continuation_prompt") or "Continue the story to bridge this gap. End of text before the gap:\n\n{before}\n\nText that must follow after your bridge:\n\n{after}"
        prompt = fill_tpl.replace("{before}", (edited[-1] if edited else before)[-1500:]).replace("{after}", after[:1500])
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content":prompt}]
        full = ""
        async for piece, _think in stream_llm(conn, msgs, model, temperature=config.get("temperature",0.5), num_ctx=config.get("num_ctx",8192), num_predict=config.get("num_ctx",8192)): full += piece
        new_content, gap_remaining = "\n\n".join(edited) + "\n\n" + full.strip() + "\n\n" + after, False
    elif marker in content:
        open_tpl = config.get("open_ended_prompt") or "Continue this toward its stated goal. Existing text ends:\n\n{tail}"
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content": open_tpl.replace("{tail}", (edited[-1] if edited else before)[-1500:])}]
        full = ""
        async for piece, _think in stream_llm(conn, msgs, model, temperature=config.get("temperature",0.5), num_ctx=config.get("num_ctx",8192), num_predict=config.get("num_ctx",8192)): full += piece
        new_content, gap_remaining = "\n\n".join(edited) + "\n\n" + full.strip(), True
    else:
        new_content, gap_remaining = "\n\n".join(edited) or content, False
    return {"text": new_content, "gap_remaining": gap_remaining}

async def chunked_file_pass(config: dict, ctx) -> dict:
    conn = get_conn(config.get("conn_id", ""))
    model = config.get("model", "")
    if not conn or not model: raise RuntimeError("chunked_file_pass: connection/model not configured")
    num_ctx = int(config.get("model_ctx", 32768))
    chunk_tokens = int(config.get("chunk_tokens", 6000))
    sys_p = ctx.resolve(config.get("system_prompt", ""))
    tpl = config.get("user_template", "") or "{chunk_content}"
    sep = config.get("output_separator") or "\n\n---\n\n"
    bi = ENV["tools"]["built_ins"]
    fm_root = config.get("fm_root") or "./data/_common"
    fm = bi.FileManager(fm_root)
    files_list = ctx.scratch.get(config.get("items_source", "files"), [])
    if not isinstance(files_list, list): files_list = [files_list]
    accumulated_output = ""
    for fi, rel in enumerate(files_list):
        try: text = fm.read(rel)
        except Exception: continue
        chunks = _chunk_text(text, chunk_tokens)
        for ci, chunk in enumerate(chunks):
            user_msg = tpl.replace("{file_name}", Path(rel).name).replace("{file_path}", rel).replace("{chunk_number}", str(ci+1)).replace("{chunks_total}", str(len(chunks))).replace("{chunk_content}", chunk)
            msgs = ([] if not sys_p else [{"role": "system", "content": sys_p}]) + [{"role": "user", "content": user_msg}]
            full = ""
            async for text_piece, _think in stream_llm(conn, msgs, model, temperature=config.get("temperature", 0.3), num_ctx=num_ctx):
                if text_piece:
                    full += text_piece
                    await ctx.stream("chunk_pass", text_piece)
            if full: accumulated_output += f"\n\n<!-- file_pass | {rel} chunk {ci+1}/{len(chunks)} -->\n{full.strip()}{sep}"
            await ctx.push("file_pass_progress", {"file": rel, "chunk": ci+1, "chunks_total": len(chunks), "file_index": fi+1, "files_total": len(files_list)})
    return {"text": accumulated_output.strip(), "files_processed": len(files_list)}

async def chunked_synthesis(config: dict, ctx) -> dict:
    conn = get_conn(config.get("conn_id", "")); model = config.get("model", "")
    if not conn or not model: raise RuntimeError("chunked_synthesis: connection/model not configured")
    num_ctx, chunk_tokens = int(config.get("model_ctx", 32768)), int(config.get("chunk_tokens", 6000))
    sys_p, tpl, sep = ctx.resolve(config.get("system_prompt", "")), config.get("user_template", "") or "{chunk_content}", config.get("output_separator") or "\n\n---\n\n"
    content = str(ctx.scratch.get(config.get("source_key", "input"), "")).strip()
    if not content: return {"text": "", "chunks_processed": 0}
    chunks = _chunk_text(content, chunk_tokens)
    accumulated = ""
    for ci, chunk in enumerate(chunks):
        user_msg = tpl.replace("{chunk_content}", chunk).replace("{chunk_number}", str(ci+1)).replace("{chunks_total}", str(len(chunks)))
        msgs = ([{"role":"system","content":sys_p}] if sys_p else []) + [{"role":"user","content":user_msg}]
        full = ""
        async for text_piece, _think in stream_llm(conn, msgs, model, temperature=config.get("temperature", 0.3), num_ctx=num_ctx):
            if text_piece: full += text_piece; await ctx.stream("synthesis", text_piece)
        if full: accumulated += f"\n\n{full.strip()}{sep}"
        await ctx.push("synthesis_progress", {"chunk": ci+1, "chunks_total": len(chunks)})
    return {"text": accumulated.strip(), "chunks_processed": len(chunks)}

async def step_knowledge_insert_text(config: dict, ctx) -> dict:
    conn = get_conn(config.get("conn_id", ""), conn_type="lightrag")
    if not conn: raise RuntimeError("knowledge_insert_text step: no knowledge connection configured")
    text = ctx.resolve(config.get("text_template", "{input}"))
    res = await lightrag_insert_text(conn, text, config.get("source_label", ""))
    res_dict = res if isinstance(res, dict) else {"status": res}
    return res_dict

async def step_echo(config: dict, ctx) -> dict:
    value = ctx.resolve(config.get("template", "{input}"))
    await ctx.push("echo", {"value": value})
    return {"value": value, "text": value}

async def step_file_write(config: dict, ctx) -> dict:
    bi = ENV["tools"]["built_ins"]
    fm_root = config.get("fm_root") or "./data/_common"
    fm = bi.FileManager(fm_root)
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(fm_root) / "_shadow"))
    rel_path = ctx.resolve(config.get("path") or "")
    if not rel_path.strip(): raise RuntimeError("file_write: resolved path is empty - check this node's Location/filename fields")
    raw_key = str(config.get("content_key", "text")).strip()
    if not raw_key: raise RuntimeError("file_write: content_key is empty - set a scratch key or {node_id.field} reference")
    template = raw_key if raw_key.startswith("{") else "{" + raw_key + "}"
    content = ctx.resolve(template)
    if not content: raise RuntimeError(f"file_write: '{raw_key}' resolved to empty content - check Result Mapping on the upstream node, or reference it directly as {template}")
    entry = shadow.stage(rel_path, str(content), author=f"pipeline:{ctx.job_id}")
    return {"path": rel_path, "status": entry["status"]}

async def step_python_exec(config: dict, ctx) -> dict:
    if config.get("script_body"):
        tmp = Path(f"./data/ai_manager/_scratch_scripts/{ctx.job_id}_{ctx.node_id}.py")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(ctx.resolve(config["script_body"]))
        script_path = str(tmp)
    else:
        script_path = config.get("script_path", "")
    if not script_path or not Path(script_path).is_file(): raise RuntimeError(f"python_exec: script not found: {script_path}")
    arg = ctx.resolve(config.get("input_template", "{input}"))
    timeout_s = int(config.get("timeout_s", 600) or 600)
    proc = await asyncio.create_subprocess_exec(sys.executable, script_path, arg, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try: stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill(); await proc.communicate()
        raise RuntimeError(f"python_exec: timed out after {timeout_s}s")
    if proc.returncode != 0: raise RuntimeError(f"python_exec: exit {proc.returncode}\n{stderr.decode(errors='replace')[-1000:]}")
    out = stdout.decode(errors="replace").strip()
    try: parsed = json.loads(out.splitlines()[-1]) if out else {}
    except Exception: parsed = {"raw_stdout": out}
    return parsed if isinstance(parsed, dict) else {"result": parsed}

async def step_file_write_binary(config: dict, ctx) -> dict:
    bi = ENV["tools"]["built_ins"]
    fm_root = config.get("fm_root") or "./data/_common"
    fm = bi.FileManager(fm_root)
    shadow = bi.ShadowStore(fm, config.get("shadow_dir") or (Path(fm_root) / "_shadow"))
    src = ctx.scratch.get(config.get("content_key", "image_file"), {})
    src_filename = src.get("file_name") if isinstance(src, dict) else str(src)
    source_path = Path(config.get("source_root") or ".") / src_filename
    if not source_path.is_file(): raise RuntimeError(f"file_write_binary: source not found: {source_path}")
    rel_path = ctx.resolve(config.get("path") or "")
    if not rel_path.strip(): raise RuntimeError("file_write_binary: resolved path is empty - check this node's Location/filename fields")
    entry = shadow.stage_binary(rel_path, source_path.read_bytes(), author=f"pipeline:{ctx.job_id}")
    return {"path": rel_path, "status": entry["status"]}

async def step_call_pipeline(config: dict, ctx) -> dict:
    pid = config.get("pipeline_id", "")
    if not pid: raise RuntimeError("call_pipeline: no pipeline selected")
    depth = int(config.get("_call_depth", 0))
    scratch = await engine.run_inline(ctx.username, pid, inputs={"input": ctx.resolve(config.get("input_template") or "{input}"), **_resolve_extra_inputs(config, ctx)}, depth=depth)
    return scratch if isinstance(scratch, dict) else {"scratch": scratch}

async def step_foreach_call_pipeline(config: dict, ctx) -> dict:
    raw = ctx.scratch.get(config.get("items_source", ""), config.get("items_source", ""))
    if isinstance(raw, str):
        try: items = json.loads(raw)
        except Exception: items = [x.strip() for x in raw.split("\n") if x.strip()]
    else: items = raw if isinstance(raw, list) else []
    if not items: raise RuntimeError("foreach_call_pipeline: items_source resolved to an empty list")
    pid = config.get("pipeline_id", "")
    if not pid: raise RuntimeError("foreach_call_pipeline: no pipeline selected")
    depth = int(config.get("_call_depth", 0))
    results = []
    for i, item in enumerate(items):
        await ctx.progress(f"item {i+1}/{len(items)}: {str(item)[:60]}")
        results.append(await engine.run_inline(ctx.username, pid, inputs={"input": str(item), **_resolve_extra_inputs(config, ctx)}, depth=depth))
    return {"results": results, "count": len(results)}

async def step_branch_on(config: dict, ctx) -> dict:
    key = config.get("decision_key", "decision")
    value = str(ctx.scratch.get(key, ""))
    try: routes = json.loads(config.get("routes_json", "{}") or "{}") if isinstance(config.get("routes_json"), str) else config.get("routes_json", {})
    except Exception: routes = {}
    pid = routes.get(value) or config.get("default_pipeline_id", "")
    if not pid: raise RuntimeError(f"branch_on: no route configured for decision value '{value}'")
    depth = int(config.get("_call_depth", 0))
    scratch = await engine.run_inline(ctx.username, pid, inputs={"input": ctx.resolve(config.get("input_template") or "{input}"), **_resolve_extra_inputs(config, ctx)}, depth=depth)
    return scratch if isinstance(scratch, dict) else {"scratch": scratch}

async def step_format_each(config: dict, ctx) -> dict:
    """Turns a list of dicts (typically foreach_call_pipeline's 'results') into repeated text blocks.
    item_template uses {field} for keys INSIDE one item directly
    - not {node.field}, since a list item has no node id of its own, just whatever keys that sub-run's scratch ended up with (its "input", plus any top-level keys its own steps wrote via result_map)."""
    items = ctx.scratch.get(config.get("items_key",""), [])
    if not isinstance(items, list): raise RuntimeError("format_each: items_key must resolve to a list")
    tpl, sep = config.get("item_template", "{input}"), config.get("separator", "\n\n---\n\n")
    def _fmt(item):
        out = tpl
        if isinstance(item, dict):
            for k, v in item.items(): out = out.replace("{" + k + "}", str(v))
        return out
    return {"text": sep.join(_fmt(i) for i in items), "count": len(items)}
