# Memory Brain Agent — Implementation Progress

## Phase 1: Low-Level Brain Tools ✅
- Created `openmemory/api/app/brain/tools.py` with 8 primitive tools
- Created `brain_audit` SQLite table (auto-created on import)
- All tools reuse existing connection logic (mem0 client, Neo4j driver, SQLite engine)
- Read-only tools validate against mutating keywords
- Write tools log all mutations to audit table
- `vector_store` includes dedup check (cosine >= 0.85) and async graph extraction

## Phase 2: Brain Agent Loop ✅
- Created `openmemory/api/app/brain/agent.py` with `MemoryBrainAgent` class
- Created `openmemory/api/app/brain/prompts.py` with system prompt and JSON schema
- Agent uses Ollama JSON mode (not native tool_calls) for reliable structured output
- JSON parse failure retry logic (up to 2 retries with corrective prompt)
- Dry-run mode for confirmation flow (collects destructive actions without executing)
- `run_confirmed()` for executing previously planned actions
- Module-level singleton `brain_agent`

## Phase 3: REST API Endpoints ✅
- Created `openmemory/api/app/routers/brain.py` with 4 endpoints
- `POST /api/v1/brain/ask` — read-only brain query
- `POST /api/v1/brain/do` — read-write with confirmation gate
- `GET /api/v1/brain/audit` — audit log viewer
- `GET /api/v1/brain/status` — brain config status
- Registered in `main.py` and `routers/__init__.py`

## Phase 4: MCP Tools ✅
- Added `memory_ask` and `memory_do` to `mcp_server.py` (SSE transport)
- Added `memory_ask` and `memory_do` to `mcp-server/server.js` (stdio transport)
- All 9 existing MCP tools preserved (7 original + 2 new)

## Phase 5: Documentation & Testing ✅
- Created `openmemory/api/app/brain/README.md` — architecture, tools, config
- Created `PROGRESS.md` (this file) and `DECISIONS.md`
- Created test suite: `tests/test_brain_tools.py`, `tests/test_brain_agent.py`, `tests/test_brain_api.py`

## Pending
- Build container and run integration tests
- Verify UI still works
- Verify all v1 MCP tools still respond
