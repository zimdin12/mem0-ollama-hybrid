# Fork Changes: Ollama + Hybrid Memory (Local-Only Mode)

All changes in this fork (`mem0-ollama-hybrid`) relative to upstream
[mem0ai/mem0](https://github.com/mem0ai/mem0).

**Goal**: Run OpenMemory fully locally using Ollama (no OpenAI dependency), with hybrid
memory mode enabled (vector store + graph database + temporal context).

---

## 1. JSON-Based Graph Extraction (`openmemory/api/fix_graph_entity_parsing.py`)

**Problem**: Upstream uses 3-4 sequential LLM calls with tool/function calling for graph
extraction (entity extraction → relationship extraction → delete decision). qwen3:4b cannot
reliably produce tool_calls — returns empty arrays, wrong arguments, or malformed JSON.

**Solution**: Complete replacement with a single Ollama JSON mode call:
- Uses `format: 'json'` in the Ollama API (forces valid JSON output)
- Single prompt extracts both entities and relationships in one call
- Entity blocklist filters file extensions (.php, .js) and generic words
- Skips the delete step entirely (too unreliable with small LLMs)
- Validates relationships (both source and target must exist in entity map)
- Filters self-referential relationships

**Result**: 79-84 graph nodes and 109-135 relationships from test data (was 19 nodes with
tool calling). Graph contributes to 7/10 search queries.

## 2. Ollama LLM Compatibility Fix (`openmemory/api/app/utils/memory.py`)

**Problem**: mem0 expects LLM responses in specific formats (tool_calls dicts for graph
extraction, plain strings for fact extraction). Small local models like qwen3:4b return
responses in inconsistent formats.

**Solution**: `_apply_essential_ollama_fix()` monkey-patches `OllamaLLM.generate_response()`:
- Returns **strings** for fact extraction (fixes `AttributeError: 'dict' has no 'strip'`)
- Returns **dicts with tool_calls** for entity/relationship extraction
- Parses multiple qwen3 output formats: function call syntax, bare JSON arrays, structured
  text blocks, bullet lists, mixed format arrays

## 3. Enhanced Memory Manager (`openmemory/api/app/utils/enhanced_memory.py`)

**New class** providing hybrid intelligence across all storage layers:

### Hybrid Search (3 dimensions)
1. **Vector search**: Semantic similarity in Qdrant
2. **Graph search**: Entity relationships in Neo4j via directed Cypher queries
3. **Temporal search**: Recency-weighted (30-day decay) from SQLite

Results are deduplicated by ID/content, then interleaved by source:
- 60% vector (most informative for content retrieval)
- 30% graph (structural relationships, entity queries)
- 10% temporal (recent context)

Graph results scored at 0.65 (vs vector 0.5-0.85) to provide context without dominating.

### Smart Addition
- Extracts facts via regex with protected splitting (file extensions, version numbers, paths)
- Checks each fact against Qdrant for semantic duplicates (threshold ≥ 0.85)
- Stores each fact individually with `infer=False, graph=False` (no LLM re-extraction)
- Runs ONE graph extraction call on the full original text in a **background thread**
- Returns detailed status: new / updated / duplicate

### Fact Extraction Improvements
Regex-based splitting that protects:
- File extensions: `.php`, `.js`, `.py`, `.json`, etc.
- Version numbers: `v3.3`, `PHP 8.2+`, `1.17.0`
- File paths: `src/components/App.tsx`
- Uses placeholder substitution during split, restores after
- Post-validation: minimum 20 chars, rejects orphaned fragments

**Result**: 0 broken fragments (was ~28% with naive splitting on code content).

### Async Graph Extraction
Graph extraction (~3s LLM call) runs in a daemon thread. The API responds immediately
after vector storage (~0.5s). Graph populates within 3-5 seconds asynchronously.

**Result**: 5x faster memory addition (0.5s vs 4.5s for medium texts).

## 4. Custom Prompts for Small Models (`openmemory/api/custom_update_prompt.py`)

Default mem0 prompts are designed for GPT-4 and are too verbose for 4B models.

Three optimized prompts:
- **Fact extraction**: Forces JSON-only output, many examples of breaking prose into
  individual facts, today's date for context
- **Update memory**: Clear decision rules (ADD/UPDATE/DELETE/NONE) with bias toward
  preserving information rather than over-deduplicating
- **Graph relationships**: Code-aware relationship types (is_a, uses, extends, contains,
  implements, depends_on), filters file extensions and generic words

## 5. Core mem0 Library Changes

### `mem0/memory/main.py`
- Added `graph: bool = True` parameter to `Memory.add()`
- When `graph=False`, skips graph extraction in the ThreadPoolExecutor
- Allows per-fact vector storage without redundant per-fact graph calls

### `mem0/memory/graph_memory.py`
- **Entity list bug fix**: When `custom_prompt` was set, the user message omitted the
  entity list, causing relationship extraction to fail. Now always includes entity list.
- Added tech-specific entity type rules to the entity extraction system prompt

## 6. MCP Server Enhancements (`openmemory/api/app/mcp_server.py`)

### Tools (7 total)

| Tool | Description |
|------|-------------|
| `add_memories` | Smart add with dedup (via `enhanced_memory_manager`) |
| `search_memory` | Hybrid search: vector + graph + temporal (limit: 10) |
| `list_memories` | List all memories with permission filtering |
| `delete_memories` | Delete by ID, syncs SQLite state |
| `delete_all_memories` | Bulk deletion with state management |
| `handle_conversation` | Process user message + LLM response, extract memorable content |
| `get_related_memories` | Relationship traversal, grouped by source (vector/graph/temporal) |

### Permission Filter Fix
Graph and temporal results have generated UUIDs (not in SQLite). The permission filter
checked `result.id in accessible_memory_ids`, silently dropping all non-vector results.
Fixed: graph/temporal results bypass permission check (they have no SQLite entry).

### Entity Extraction for Search
Improved query entity extraction: strips punctuation, expanded stopwords list. Prevents
issues like `"steven?"` not matching `"steven"` in Neo4j CONTAINS queries.

## 7. Config Loading Hierarchy (`openmemory/api/app/utils/memory.py`)

Upstream loads config from database only. This fork uses a cascade:

1. **Defaults** from `get_default_memory_config()` (auto-detects Ollama URL in Docker)
2. **config.json** overrides (production settings with `env:VAR_NAME` references)
3. **Database** — IGNORED for llm/embedder/vector_store (prevents OpenAI defaults)
   - Only `custom_instructions` read from DB
4. **Environment variables** parsed throughout (type conversion for ports and dimensions)

### Docker Host Resolution
Auto-detects Ollama URL when running in Docker:
1. `OLLAMA_HOST` env var
2. `host.docker.internal` (Docker Desktop Mac/Windows)
3. Docker bridge gateway from `/proc/net/route` (Linux)
4. Fallback: `172.17.0.1`

## 8. Config Files (`openmemory/api/config.json`)

All provider fields use `env:VARIABLE_NAME` pattern:
- `LLM_PROVIDER`, `LLM_MODEL`, `OLLAMA_BASE_URL`
- `EMBEDDER_PROVIDER`, `EMBEDDER_MODEL`, `EMBEDDER_OLLAMA_BASE_URL`
- `VECTOR_STORE_PROVIDER`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

Added `graph_store` section (Neo4j) and `vector_store` section (Qdrant) which were
absent from upstream config.

## 9. Categorization (`openmemory/api/app/utils/categorization.py`)

Upstream uses hardcoded `OpenAI()` client with `gpt-4o-mini` and structured output
(`beta.chat.completions.parse`).

Fork uses:
- `OpenAI(base_url=OLLAMA_BASE_URL/v1, api_key="ollama")` compatible client
- Regular `chat.completions.create` (no structured output)
- Manual JSON parsing with markdown code block fallback
- Returns empty list on failure (non-fatal)

## 10. Memory Router (`openmemory/api/app/routers/memories.py`)

### New Graph Endpoints
- `GET /graph/stats` — Entity/relationship counts
- `GET /{memory_id}/entities` — Extract entities from a specific memory
- `GET /graph/search` — Search for entities by name
- `GET /graph/entity/{entity_name}/related` — Related entities via Cypher
- `GET /graph/entity/{entity_name}/memories` — Memories mentioning an entity
- `GET /graph/visualize` — Graph nodes/edges for visualization
- `GET /graph/context/{topic}` — Comprehensive context via graph traversal

### Memory Operations
- `CreateMemoryRequest` accepts `messages[]` (conversation format)
- DELETE events from mem0 sync to SQLite state (no phantom entries)
- `POST /sync-storage` — Bi-directional sync between Qdrant and SQLite

## 11. Database (`openmemory/api/app/database.py`)

SQLite persistence fix:
```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/openmemory.db")
os.makedirs("db", exist_ok=True)
```
Ensures `./db/` directory exists for Docker volume mounting (`openmemory-db` volume).

## 12. Docker Compose Override (`openmemory/docker-compose.override.yml`)

- **Neo4j 5.26.4**: Graph database with APOC plugin, ports 8474/8687, healthcheck
- **Qdrant init**: Pre-creates `openmemory` collection (1024-dim cosine)
- `extra_hosts: host.docker.internal:host-gateway` for Ollama access from containers
- Shared `openmemory_network` bridge network
- Service dependency ordering with healthchecks

## 13. Qdrant Init Script (`openmemory/init-qdrant.sh`)

Waits for Qdrant readiness, then creates `openmemory` collection with 1024-dim Cosine
vectors. Lets mem0 auto-create `mem0migrations` collection. Non-fatal if collection exists.

## 14. Requirements Changes

### `openmemory/api/requirements.txt`
- Added `langchain-neo4j` — Required for mem0's Neo4j graph store
- Added `rank-bm25` — BM25 text search for hybrid search

## 15. Test Suite (`test_memory_system.py`)

5-phase comprehensive test:
1. **Insert**: 12 chunks of varied sizes (short/medium/long, dense/sparse, + duplicate test)
2. **Validate**: Verify Qdrant vectors, Neo4j nodes/rels, SQLite memories all match
3. **Graph quality**: List all entities/relationships, check for garbage (file extensions)
4. **Search quality**: 10 diverse queries, verify correct top result, measure graph contribution
5. **Dedup quality**: Check for exact duplicates and broken fragments in Qdrant

---

## Deduplication Pipeline

Three layers prevent duplicate storage:

1. **main.py** (cosine ≥ 0.95): mem0's built-in dedup in `_add_to_vector_store`
2. **enhanced_memory.py** (cosine ≥ 0.85): Pre-check in `_find_novel_facts` before any storage
3. **infer=False**: Skip LLM re-extraction when storing pre-extracted facts

---

## Files Summary

### Added (not in upstream)

| File | Lines | Purpose |
|------|-------|---------|
| `openmemory/api/app/utils/enhanced_memory.py` | ~570 | Hybrid search, smart add, async graph |
| `openmemory/api/fix_graph_entity_parsing.py` | ~225 | JSON-based graph extraction |
| `openmemory/api/custom_update_prompt.py` | ~140 | Optimized prompts for qwen3:4b |
| `openmemory/docker-compose.override.yml` | ~50 | Neo4j + networking |
| `openmemory/init-qdrant.sh` | ~20 | Collection initialization |
| `mcp-server/server.js` | ~200 | Standalone MCP server (stdio) |
| `mcp-server/SKILL.md` | ~50 | Claude Code skill definition |
| `test_memory_system.py` | ~300 | Comprehensive test suite |

### Modified (vs upstream)

| File | Key Changes |
|------|-------------|
| `openmemory/api/app/utils/memory.py` | Ollama fix (~300 lines), config cascade, Docker host detection |
| `openmemory/api/app/utils/categorization.py` | OpenAI → Ollama with JSON fallback parsing |
| `openmemory/api/app/mcp_server.py` | Enhanced memory manager, 7 tools, permission filter fix |
| `openmemory/api/app/routers/memories.py` | 7 graph endpoints, DELETE sync, messages[] support |
| `openmemory/api/app/routers/config.py` | Env-based defaults, no auto-create DB rows |
| `openmemory/api/app/database.py` | SQLite path for Docker volume persistence |
| `openmemory/api/config.json` | Env var references for all providers |
| `mem0/memory/main.py` | `graph=False` parameter on `Memory.add()` |
| `mem0/memory/graph_memory.py` | Entity list bug fix, tech entity types |
| `openmemory/api/requirements.txt` | +langchain-neo4j, +rank-bm25 |
