# ai_manager (tool)

Centralized AI infrastructure for Portal Server: connections, the provider-agnostic wire-format layer, the pipeline execution engine, and the universal-node step registry. Modules (Athena, Kimi, Tessa, Image) call into this tool directly - they never hold their own connections or run their own execution loop.

## Layout

- `connections.py` - per-connection-type wire format, read at runtime from JSON profiles (`ollama.json`, `lightrag.json`, `flux2_text.json`, `flux2_image.json`). Adding a new backend is a JSON file drop, not a code change.
- `engine.py` - wave-based async DAG execution over a `Flow` graph. Nodes run once every pipeline key they need is present in the job's shared data object; no explicit prev/next edges.
- `steps.py` - the universal node type registry (`generate`, `transform`, `knowledge`, `file_read`, `file_write`, `pipeline`, `pipeline_foreach`, `branch`, `file_list`).
- `flow.py` - `Flow`/`FlowNode` data classes, JSON (de)serialization, causal-level display helper for the graphical builder.
- `resources.py` - CNode registry and resource-pool resolution (tag-based connection selection across multiple machines).

## Adding a connection type

Drop a new JSON profile in this directory declaring `endpoints`, `options_schema`, and `response_parser`. No code in `connections.py` needs to change for a standard chat-completion-shaped API.

## Adding a node type

Register via `steps.register_node_type(name, fn, label, in_keys, out_keys, config_schema, guide)`. `fn` is `async def fn(config: dict, data: dict, ctx: NodeContext) -> dict`, returning a dict of logical output names - the engine remaps these through the node's own `key_map` before merging into the shared data object.
https://polyformproject.org/licenses/noncommercial/1.0.0
## License

Licensed under the [Polyform Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). See `FSEP.md` at the repository root for commercial use terms.