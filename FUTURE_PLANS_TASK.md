# Memory Agent (v2) — Task Reference

## Quick Reference

| Item | Value |
|------|-------|
| Branch | `memory_agent` |
| Model | `LLM_MODEL` env var (same as extraction, default: qwen3.5:4b) |
| Ollama URL | `OLLAMA_BASE_URL` env var (default: http://ollama:11434) |
| Max steps | `BRAIN_MAX_STEPS` env var (default: 12) |
| REST endpoint | `POST /api/v1/brain` |
| MCP tool | `memory_agent` |
| Tests | `tests/test_brain_tools.py`, `tests/test_brain_agent.py`, `tests/test_brain_api.py` |

## Key Design Decisions

1. **Single endpoint** — no ask/do split. Agent determines intent from the request.
2. **Single model** — no separate BRAIN_LLM_MODEL. Uses LLM_MODEL + OLLAMA_BASE_URL.
3. **Full access** — no read-only mode. The agent has all 8 tools available always.
4. **v1 coexists** — all 7 v1 MCP tools remain available alongside `memory_agent`.
5. **JSON mode** — Ollama `format: "json"` for structured output (more reliable than native tool_calls).

## Testing

```bash
# Phase 1: Brain tools (12 tests, runs inside container)
docker exec -e PYTHONPATH=/usr/src/openmemory openmemory-mcp \
  python3 /usr/src/openmemory/tests/test_brain_tools.py

# Phase 2: Agent loop (5 tests, runs inside container)
docker exec -e PYTHONPATH=/usr/src/openmemory openmemory-mcp \
  python3 /usr/src/openmemory/tests/test_brain_agent.py

# Phase 3: REST API (5 tests, runs from host)
python mem0-fork/tests/test_brain_api.py
```

## Files

```
openmemory/api/app/brain/
├── __init__.py        # Package marker
├── agent.py           # MemoryBrainAgent class + BrainResult
├── prompts.py         # System prompt (single prompt, no read-only split)
├── tools.py           # 8 DB tools + audit table
└── README.md          # Architecture docs

openmemory/api/app/routers/brain.py   # POST /brain, GET /brain/audit, /brain/status
openmemory/api/app/mcp_server.py      # memory_agent MCP tool
mcp-server/server.js                   # memory_agent Node.js MCP tool
tests/test_brain_*.py                  # 3 test suites
```
