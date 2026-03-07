# Memory Brain Agent — Architectural Decisions

## D1: Brain Model — qwen3.5:9b on Ollama

**Decision**: Use `qwen3.5:9b` on the same Ollama instance as the extraction pipeline.

**Rationale**:
- Benchmarked as solid all-rounder with perfect dedup and config-insensitive extraction
- Better reasoning than 4b for multi-step agent loops
- ~6GB VRAM — fits alongside extraction model (qwen3.5:4b, ~2.7GB) and embedding model
  (qwen3-embedding:0.6b, ~0.5GB) on RTX 4090 (24GB)
- Same Ollama API — no need for a separate LM Studio connection

**Alternatives considered**:
- qwen3.5:4b — lighter but may struggle with multi-step reasoning
- LM Studio models (qwen3.5-35b-a3b, llama-3.1-8b) — requires cross-host HTTP, added complexity

## D2: JSON Mode over Native Tool Calls

**Decision**: Use Ollama's `format: 'json'` with a structured schema, not native `tool_calls`.

**Rationale**:
- qwen3 models have unreliable multi-step native tool calling (tested extensively in v1)
- JSON mode forces valid JSON output every turn
- The schema (thinking/action/args/final) is simple enough for any 4B+ model
- Easier to debug — every turn is a visible JSON object
- Same approach that works reliably for graph extraction in v1

## D3: Separate Brain Router (not extending memories.py)

**Decision**: New router `app/routers/brain.py` instead of adding endpoints to `memories.py`.

**Rationale**:
- `memories.py` is already 500+ lines with many endpoints
- Clean separation of concerns — brain is a new feature layer, not a memory CRUD extension
- Easier to disable/remove the brain feature without touching existing code
- Independent testing

## D4: Confirmation Gate for Destructive Operations

**Decision**: `BRAIN_CONFIRM_DESTRUCTIVE=true` (default) requires a two-step flow for deletes via REST.

**Rationale**:
- Safety: prevents accidental bulk deletion
- MCP tools auto-confirm (the calling LLM has already decided)
- REST API callers get a preview of what will be deleted before executing
- Configurable via env var for automated pipelines

## D5: Sync Tool Functions (not async)

**Decision**: Brain tools are sync functions, not async.

**Rationale**:
- All underlying clients (mem0, Neo4j driver, SQLite) are sync
- Adding async wrappers would just add `run_in_executor` boilerplate
- The agent loop is sequential (one tool at a time) — no parallelism benefit
- The brain agent `run()` is sync; the FastAPI endpoint wraps it

## D6: Reuse Existing Connection Logic

**Decision**: Brain tools get database connections via `get_memory_client()`, `engine`, and
the existing Neo4j driver access pattern from `enhanced_memory.py`.

**Rationale**:
- No connection duplication
- Benefits from existing fixes (Ollama compatibility, Docker host detection)
- Single point of configuration (env vars)

## D7: No New Python Dependencies

**Decision**: No new packages added to requirements.txt.

**Rationale**:
- The brain uses only `requests` (already available), `json`, and existing database clients
- Ollama is called via raw HTTP (same as graph extraction)
- Kept the dependency footprint minimal
