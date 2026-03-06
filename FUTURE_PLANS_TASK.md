# Claude Code Task: Memory Brain Agent (v2)

## Your Mission

You are implementing the **Memory Brain Agent** described in `FUTURE_PLANS.md` on a new
branch called `memory_agent`. This is a long autonomous task. Read this entire document
before writing a single line of code.

---

## Autonomy Rules (READ FIRST)

- **Do not stop to ask questions.** Make decisions and document them in code comments or
  a `DECISIONS.md` file in the repo root.
- **Do not wait for confirmation** before creating files, running tests, or making
  architectural decisions.
- If you hit an ambiguity, pick the more conservative/backward-compatible option and note it.
- If a tool call or test fails, debug it yourself. Retry up to 3 times with different
  approaches before moving on and leaving a `# TODO: failed, reason:` comment.
- After completing each major phase, run the relevant tests and fix failures before moving
  to the next phase. Do not skip testing.
- **Keep a running `PROGRESS.md`** in the repo root. Update it after every phase so
  progress is visible even if the session is interrupted.

---

## Environment

| Service         | URL / Port                          | Notes                                      |
|-----------------|-------------------------------------|--------------------------------------------|
| Ollama (extraction)| `http://ollama:11434` (inside container) / `http://localhost:11435` (host) | Has `qwen3:4b-instruct-2507-q4_K_M` + `qwen3-embedding:0.6b` — for embeddings and fact extraction. List models: `GET http://localhost:11435/api/tags` |
| LM Studio (brain) | `http://host.docker.internal:11434` (inside container) / `http://localhost:11434` (host) | Multiple models available — use for the brain agent loop. List models: `GET http://localhost:11434/v1/models` (OpenAI-compatible). LM Studio does NOT support Ollama-style `/api/chat` — use OpenAI-compatible `/v1/chat/completions` instead. |
| Qdrant          | `http://localhost:6333` (host) or `http://mem0_store:6333` (container) | |
| Neo4j Bolt      | `bolt://localhost:7687` (host) or `bolt://neo4j:7687` (container) | user: neo4j / pass: openmemory |
| OpenMemory API  | `http://localhost:8765`             | The existing FastAPI service               |
| OpenMemory UI   | `http://localhost:3100`             | Keep working                               |

**100% local. No OpenAI, no Anthropic, no cloud API calls anywhere in the codebase.**

The embedding model is `qwen3-embedding:0.6b` at **1024 dimensions** (on Ollama). Do not change this.
The extraction LLM is `qwen3:4b-instruct-2507-q4_K_M` (on Ollama) — keep using this for fact/graph extraction.

### Brain Agent Model Selection

LM Studio has multiple models loaded. **Test which one works best for multi-step tool calling.**
List them: `GET http://localhost:11434/v1/models`

Models available as of Mar 2026:
- `qwen/qwen3.5-35b-a3b` — 35B MoE (3B active params), likely best reasoning for agent loops
- `qwen/qwen3-coder-next` — code-focused, good at structured JSON output
- `meta-llama-3.1-8b-instruct` — reliable tool caller, well-tested
- `zai-org/glm-4.7-flash` — fast
- `mistralai/devstral-small-2-2512` — code-focused
- `huihui-ai_huihui-gpt-oss-20b-abliterated` — uncensored 20B

**Strategy**:
1. Try `qwen3.5:4b` on Ollama first — same backend as extraction, simplest setup, and qwen3.5 is a significant reasoning upgrade over qwen3 at the same size
2. If 4B isn't reliable enough for multi-step agent loops, try `qwen/qwen3.5-35b-a3b` on LM Studio (35B MoE, only 3B active — best reasoning)
3. Fall back to `meta-llama-3.1-8b-instruct` on LM Studio if qwen models struggle with tool calling
4. Document the model you chose and why in DECISIONS.md

**IMPORTANT**: LM Studio uses OpenAI-compatible API, NOT Ollama-style API:
- Endpoint: `http://host.docker.internal:11434/v1/chat/completions`
- Use `openai.OpenAI(base_url="http://host.docker.internal:11434/v1", api_key="lm-studio")`
- For JSON mode: set `response_format={"type": "json_object"}` in the chat completion call
- Do NOT use Ollama's `format: 'json'` or `/api/chat` — those don't work with LM Studio

**Ollama also has `qwen3.5:4b`** (or will soon) — this is the simplest option since it uses
the same Ollama API as the extraction pipeline. qwen3.5 has better reasoning than qwen3 at
the same parameter count, so it may handle multi-step agent loops that qwen3:4b couldn't.
If using Ollama for the brain: `http://ollama:11434` (inside container), same `/api/chat` endpoint.

---

## Step 0: Setup

```bash
git checkout main
git pull
git checkout -b memory_agent
```

Create `PROGRESS.md` and `DECISIONS.md` in the repo root immediately.

Then **read the codebase** before touching anything:
- `FUTURE_PLANS.md` — the v2 architecture spec (your primary source of truth)
- `FORK_CHANGES.md` — what was already customized
- `TESTING_GUIDE.md` — the existing test protocol
- `openmemory/api/app/utils/enhanced_memory.py` — the main custom logic
- `openmemory/api/app/utils/memory.py` — Ollama compatibility layer
- `openmemory/api/app/mcp_server.py` — current 7 MCP tools
- `openmemory/api/app/routers/memories.py` — REST API routers
- `mem0/memory/main.py` and `mem0/memory/graph_memory.py` — core mem0 logic

---

## Phase 1: Low-Level Brain Tools (the DB access layer)

Create `openmemory/api/app/brain/tools.py`.

This module exposes the 8 primitive tools the brain agent will use. All are plain Python
async functions (no LLM involvement). They talk directly to Qdrant, Neo4j, and SQLite.
Reuse existing connection logic from `enhanced_memory.py` and `memory.py` — do not
duplicate connection setup.

```python
# Required function signatures (implement these):

async def vector_search(query: str, limit: int = 10, user_id: str = None) -> list[dict]
    # Embed query via Ollama, search Qdrant, return [{id, text, score}, ...]

async def vector_store(text: str, user_id: str, memory_id: str = None) -> str
    # Embed text via Ollama, store in Qdrant, return stored ID
    # IMPORTANT: Also trigger async graph extraction (reuse enhanced_memory._background_graph_extract)
    # so facts stored via the brain also appear in the knowledge graph

async def vector_delete(ids: list[str]) -> int
    # Delete vectors from Qdrant by ID, return count deleted

async def graph_query(cypher: str, user_id: str = None) -> list[dict]
    # Execute read-only Cypher on Neo4j, return rows as dicts
    # Raise ValueError if query contains mutating keywords (CREATE/MERGE/DELETE/SET/REMOVE)

async def graph_mutate(cypher: str, user_id: str = None) -> int
    # Execute write Cypher on Neo4j, return affected count
    # Log every mutation to SQLite audit table (brain_audit)

async def sql_query(sql: str, params: dict = None) -> list[dict]
    # Execute read-only SQL on SQLite, return rows as dicts
    # Raise ValueError if SQL contains mutating keywords

async def sql_mutate(sql: str, params: dict = None) -> int
    # Execute write SQL on SQLite, return affected count
    # Log to brain_audit table

async def embed(text: str) -> list[float]
    # Call Ollama embedding endpoint, return raw vector
```

Add a `brain_audit` table to SQLite (via Alembic migration or direct DDL if Alembic is
not used in this project — check first):

```sql
CREATE TABLE IF NOT EXISTS brain_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    tool TEXT NOT NULL,           -- 'graph_mutate' | 'sql_mutate' | 'vector_delete' etc.
    user_id TEXT,
    input TEXT,                   -- JSON-encoded args
    output TEXT,                  -- JSON-encoded result
    error TEXT                    -- NULL if successful
);
```

**Tests for Phase 1** (create `tests/test_brain_tools.py`):
- Each tool has at least one happy-path test
- `graph_query` raises on mutating Cypher
- `sql_query` raises on mutating SQL
- `vector_search` returns scored results
- `embed` returns a list of 1024 floats
- Run: `docker exec openmemory-mcp python -m pytest tests/test_brain_tools.py -v`

---

## Phase 2: The Brain Agent Loop

Create `openmemory/api/app/brain/agent.py`.

This is the core agentic loop. It takes a natural-language request and runs tool calls
in a loop until it has a final answer.

### Design

```python
class MemoryBrainAgent:
    """
    LLM agent that sits between callers and the three databases.
    Accepts natural language. Decides which tools to call. Returns synthesized answer.
    """
    
    MAX_STEPS = 12           # hard stop to prevent infinite loops
    MODEL = "qwen3.5:4b"                         # from env: BRAIN_LLM_MODEL (on Ollama)
    OLLAMA_URL = "http://ollama:11434"            # from env: BRAIN_OLLAMA_URL (same as extraction)
    
    async def run(self, request: str, user_id: str, context: list = None) -> BrainResult
```

### Tool Calling Strategy

**Use JSON-mode structured output, not native Ollama tool_calls.**

Why: `qwen3:4b-instruct` has unreliable multi-step native tool calling. Instead, prompt
the model to emit a JSON object each turn:

```json
{
  "thinking": "I need to find all facts about Steven's GPU first",
  "action": "vector_search",
  "args": {"query": "steven GPU graphics card", "limit": 10, "user_id": "steven"},
  "final": false
}
```

Or, when done:

```json
{
  "thinking": "I have all the information I need to answer",
  "action": null,
  "args": null,
  "final": true,
  "answer": "Steven uses an RTX 4090 with 24GB VRAM."
}
```

Parse this JSON every iteration. Execute the tool. Append the result to messages. Loop.

### System Prompt for the Brain

Write a tight system prompt in `openmemory/api/app/brain/prompts.py`. Key instructions:
- You are a memory management agent with access to 8 tools (list them with descriptions)
- Always emit valid JSON matching the schema above. No markdown, no preamble.
- For search operations: try both vector_search AND graph_query for completeness
- For delete operations: search first, confirm what you found, then delete
- For store operations: check for duplicates via vector_search before storing
- user_id must always be passed to tools that accept it
- Maximum thoroughness: if vector_search returns ambiguous results, do a follow-up search
- The `thinking` field is your internal reasoning — be explicit there

### BrainResult dataclass

```python
@dataclass
class BrainResult:
    answer: str
    steps: int
    tools_called: list[str]   # e.g. ["vector_search", "graph_query", "vector_delete"]
    user_id: str
    success: bool
    error: str | None = None
```

### Error handling

- If JSON parsing fails: retry the LLM call with a reminder to emit valid JSON (max 2 retries)
- If a tool raises: catch, append `{"error": str(e)}` as the tool result, let the brain
  decide how to proceed
- If `MAX_STEPS` reached: return partial answer with `"Reached step limit"` note
- Log every run (request, steps, tools_called, answer) to `brain_audit`

**Tests for Phase 2** (create `tests/test_brain_agent.py`):
- `run("What hobbies does steven have?", "steven")` → returns non-empty answer, used
  vector_search
- `run("Remember that steven uses VSCode", "steven")` → stores memory, returns confirmation
- `run("Delete all memories about dark mode", "steven")` → calls vector_search then
  vector_delete, returns deleted count
- Test that MAX_STEPS is respected (mock a model that loops forever)
- Test JSON parse failure retry logic (mock a model that returns garbage once then valid JSON)

---

## Phase 3: REST API Endpoints

Add new endpoints to `openmemory/api/app/routers/memories.py` (or a new router
`openmemory/api/app/routers/brain.py` — your choice, document it):

```
POST /api/v1/brain/ask
  Body: {"request": "...", "user_id": "steven"}
  Returns: BrainResult as JSON
  Mode: read-only (brain is told not to mutate in its system prompt)

POST /api/v1/brain/do  
  Body: {"request": "...", "user_id": "steven", "confirmed": false}
  Returns: BrainResult as JSON
  Mode: read-write
  If the brain's plan includes destructive operations (vector_delete, graph_mutate, sql_mutate)
  AND confirmed=false → return the plan with requires_confirmation=true and don't execute.
  Caller re-sends with confirmed=true to execute.

GET /api/v1/brain/audit?user_id=steven&limit=50
  Returns: last N brain_audit rows for that user
```

The confirmation logic for `/do`: inspect `tools_called` in the *planned* run. If any
destructive tool appears, serialize the plan (what will be done) and return it without
executing. This is a "dry-run" preview. When `confirmed=true`, execute for real.

**Update `openmemory/api/app/main.py`** to register the brain router.

**Tests for Phase 3** (create `tests/test_brain_api.py`):
- POST `/brain/ask` returns 200 with answer field
- POST `/brain/do` with destructive request + `confirmed=false` returns
  `requires_confirmation=true`
- POST `/brain/do` with `confirmed=true` actually executes
- GET `/brain/audit` returns rows

---

## Phase 4: MCP Tools

Add 2 new tools to `openmemory/api/app/mcp_server.py`:

```python
# Tool 1: memory_ask
# Description: "Ask a natural language question about stored memories. 
#               Searches vector store, graph, and SQLite. Returns synthesized answer.
#               Use for queries like: 'what hobbies does X have?', 
#               'what tools does X use?', 'summarize what you know about X'"
# Input: {"request": str, "user_id": str}
# Output: str (the brain's answer)
# Mode: read-only

# Tool 2: memory_do
# Description: "Perform a natural language memory operation: store, update, delete, or 
#               reorganize memories. Examples: 'remember that X switched to Godot',
#               'delete all memories about X', 'update X's job title to Y'.
#               Returns what was done."
# Input: {"request": str, "user_id": str}
# Output: str (confirmation of what was done)  
# Mode: read-write (auto-confirmed — the LLM calling this has already decided to do it)
```

Keep ALL existing 7 MCP tools. Do not remove or rename them. The brain tools are additive.

Also update `mcp-server/server.js` (the Node.js stdio MCP server) to add the same 2 tools
as pass-through HTTP calls to the FastAPI `/brain/ask` and `/brain/do` endpoints.

**Tests for Phase 4**:
- Both new MCP tools appear in the tools list response
- `memory_ask` returns a non-empty string
- `memory_do` with a store request returns confirmation

---

## Phase 5: Clean Up, Documentation, Final Testing

### Documentation

1. Update `README.md`:
   - Add "Memory Brain (v2)" section to the architecture diagram
   - Add new REST endpoints to the REST API section
   - Add new MCP tools to the MCP tools section
   - Add `BRAIN_LLM_MODEL` and `BRAIN_OLLAMA_URL` to env var table

2. Update `FUTURE_PLANS.md`:
   - Add an "Implementation Status" section at the top noting what was implemented
   - Mark completed items as ✅

3. Update `FORK_CHANGES.md`:
   - Add a `memory_agent branch` section listing all new files and changes

4. Create `openmemory/api/app/brain/README.md`:
   - Architecture diagram of the agent loop
   - System prompt design rationale
   - Tool reference table
   - Tuning guide (how to change model, MAX_STEPS, etc.)

### Environment Variables to Add

Add to `openmemory/api/.env.example`:
```env
# === Memory Brain Agent ===
BRAIN_LLM_MODEL=qwen3.5:4b                         # Model for the brain agent loop (on Ollama alongside extraction model)
BRAIN_OLLAMA_URL=http://ollama:11434                # Same Ollama instance (or http://host.docker.internal:11434 for LM Studio)
BRAIN_MAX_STEPS=12              # Max tool calls per brain invocation
BRAIN_CONFIRM_DESTRUCTIVE=true  # Require confirmed=true for deletes via /brain/do
```

### Full Integration Test

Run the complete TESTING_GUIDE.md checklist to ensure existing v1 functionality is
unbroken. Then run this brain-specific smoke test:

```bash
# 1. Insert fresh test data
curl -s -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Steven uses an RTX 4090 with 24GB VRAM. He builds games with Godot. His favorite editor is VSCode.","user_id":"steven_brain_test"}'

sleep 10  # wait for async graph extraction

# 2. Ask the brain a question (read-only)
curl -s -X POST http://localhost:8765/api/v1/brain/ask \
  -H "Content-Type: application/json" \
  -d '{"request":"What GPU does steven use and what does he do with it?","user_id":"steven_brain_test"}' \
  | python -m json.tool

# 3. Brain store
curl -s -X POST http://localhost:8765/api/v1/brain/do \
  -H "Content-Type: application/json" \
  -d '{"request":"Remember that steven_brain_test recently upgraded to 96GB of RAM","user_id":"steven_brain_test","confirmed":true}' \
  | python -m json.tool

# 4. Verify the store happened
curl -s -X POST http://localhost:8765/api/v1/brain/ask \
  -H "Content-Type: application/json" \
  -d '{"request":"How much RAM does steven have?","user_id":"steven_brain_test"}' \
  | python -m json.tool

# 5. Brain delete with confirmation gate
curl -s -X POST http://localhost:8765/api/v1/brain/do \
  -H "Content-Type: application/json" \
  -d '{"request":"Delete all memories for steven_brain_test","user_id":"steven_brain_test","confirmed":false}' \
  | python -m json.tool
# Expected: requires_confirmation: true, with a preview of what will be deleted

# 6. Check audit log
curl -s "http://localhost:8765/api/v1/brain/audit?user_id=steven_brain_test&limit=10" \
  | python -m json.tool

# 7. Cleanup test user
curl -s -X POST http://localhost:8765/api/v1/brain/do \
  -H "Content-Type: application/json" \
  -d '{"request":"Delete all memories for steven_brain_test","user_id":"steven_brain_test","confirmed":true}' \
  | python -m json.tool
```

All steps must produce valid JSON responses with no 5xx errors.

### Rebuild and Verify

```bash
docker compose build openmemory-mcp
docker compose up -d openmemory-mcp
sleep 15
curl -s http://localhost:8765/health
curl -s http://localhost:8765/docs  # should show new /brain routes
```

Confirm the UI still loads at port 3100. Confirm all existing v1 MCP tools still respond.

---

## File Structure After Implementation

```
openmemory/api/app/brain/
├── __init__.py
├── agent.py          # MemoryBrainAgent class + BrainResult dataclass
├── prompts.py        # System prompt and JSON schema for the agent loop  
├── tools.py          # 8 primitive async tool functions
└── README.md         # Brain architecture docs

tests/
├── test_brain_tools.py   # Phase 1 tests
├── test_brain_agent.py   # Phase 2 tests
└── test_brain_api.py     # Phase 3 tests

PROGRESS.md    # Updated after each phase
DECISIONS.md   # Architectural decisions with rationale
```

---

## Key Constraints (Never Violate)

1. **No cloud APIs** — every LLM/embedding call goes to Ollama at 11435 (or 11434 fallback)
2. **Backward compatibility** — all 7 existing MCP tools and all existing REST endpoints
   must continue to work exactly as before
3. **No new external Python dependencies** that aren't already in requirements.txt, unless
   absolutely necessary — if you add one, explain why in DECISIONS.md
4. **Embedding dimensions stay at 1024** — do not change the Qdrant collection schema
5. **The UI must still work** — don't break any endpoint the Next.js UI calls
6. **Test before moving to next phase** — do not skip the test steps

---

## Decision Guide (for ambiguities)

| Question | Decision |
|----------|----------|
| Should brain use native Ollama tool_calls or JSON-mode? | JSON-mode (more reliable with 4B models) |
| Should `/brain/do` auto-confirm or gate destructive ops? | Gate with `confirmed` flag (safer default) |
| Should brain tools go in a new router or extend memories.py? | New router `brain.py` (cleaner separation) |
| Should Node.js MCP server be updated too? | Yes — both tools as HTTP pass-through to FastAPI |
| What if qwen3:4b is too weak for the agent loop? | Default to LM Studio (host.docker.internal:11434) which has qwen3.5 or similar larger model. Fall back to qwen3:4b on Ollama only if LM Studio is down. |
| Should brain maintain session state across calls? | No — stateless like v1. Each call is independent. |

---

---

## Codebase Guidance (from the developer who built v1)

### What to KEEP — these work well and were hard-won:

| Component | Why it works | File |
|-----------|-------------|------|
| JSON-mode graph extraction | Single LLM call, dynamic entity types, hub-only sources — very reliable with 4B | `fix_graph_entity_parsing.py` |
| Enhanced memory smart_add | Per-fact embedding + dedup + context injection — 93% fact quality | `enhanced_memory.py` |
| Fuzzy self-ref filter + element-ID guard | Prevents self-loops in Neo4j from embedding merges | `fix_graph_entity_parsing.py` + `graph_memory.py` |
| Async chunked graph extraction | Background thread, ~2000 char chunks — API responds fast, graph populates async | `enhanced_memory.py` |
| Session context | Recent turns passed to extraction LLM — helps resolve "the game" → project name | `enhanced_memory.py` |
| Hybrid search interleaving | 60/30/10 vector/graph/temporal — graph provides structure, vector provides content | `enhanced_memory.py` |
| `infer=False, graph=False` per-fact storage | Prevents LLM re-extraction of already-extracted facts | `enhanced_memory.py` |

### What the brain agent should REUSE (not reimplement):

- **Embeddings**: Use `enhanced_memory.py`'s embedding logic or the Ollama embedding endpoint directly
- **Graph extraction**: When the brain stores new facts, trigger `_background_graph_extract()` — don't skip the graph
- **Dedup checks**: Use `_find_novel_facts()` before storing — the brain shouldn't create duplicates
- **Context injection**: `_inject_context()` prepends topic to orphaned facts — reuse this

### What the brain might REPLACE or SIMPLIFY:

- The brain's `memory_do("remember X switched to Godot")` could potentially replace `add_memories` for single-fact storage — but the bulk `add_memories` pipeline (regex split + batch dedup) is still better for large text dumps
- The brain's `memory_ask("what hobbies does X have?")` does cross-DB synthesis that `search_memory` can't — this is genuinely new capability
- Don't try to replace `conversation_memory` — its regex+LLM pipeline is tuned for conversation turns

### Known 4B model limitations (expect these with qwen3:4b, may be fixed with qwen3.5):

- Confuses similar domains: ChefMate (cooking app) gets mixed with cooking (hobby)
- Sometimes connects game features to person instead of project
- OAuth2 classified as "preference" instead of "technology"
- Occasionally omits the user entity from graph then references it in relationships
- Can't reliably do multi-step reasoning (why we use JSON-mode, not native tool_calls)

### Testing after changes:

Run `test_comprehensive.py` — it covers bulk insertion, conversation memory, search quality,
deduplication, graph quality audit, and qdrant vector count. Also follow `TESTING_GUIDE.md`.

```bash
docker exec openmemory-mcp python test_comprehensive.py
```

---

Good luck. Update `PROGRESS.md` regularly so work is visible even if the session ends
unexpectedly. When everything is green, do a final `git add -A && git commit -m "feat: memory brain agent v2 implementation"`.