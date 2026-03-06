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
- Filters self-referential relationships (exact + fuzzy containment check)
- 11 entity types with "concept" as last resort (was defaulting to concept for everything)
- Relationship hierarchy rules: projects connect to features/phases/tech, persons to skills
- Concrete prompt example showing expected output structure

**Result**: 32 nodes with 10 diverse entity types (only 3% "concept"), 42 relationships,
zero self-referential edges. Was: 77 nodes with 92% "concept" type and self-ref loops.

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
- **Context injection**: Detects project/topic name from text opening, prepends to orphaned facts
  (e.g., `"Total development time 18 months"` → `"Echoes of the Fallen: Total development time 18 months"`)
  - Only activates for texts > 300 chars (short texts are self-contained)
  - Scans only first 80 chars / first sentence for topic (avoids picking list items as topics)
  - Strips trailing prepositions from topic candidates ("Docker on Windows with" → rejected)
  - Tech name blocklist prevents tools/frameworks being used as document topics
  - Facts with existing multi-word proper nouns (e.g., "Unreal Engine") keep their own context
  - No hardcoded user/entity names — uses generic proper noun detection
- Checks each fact against Qdrant for semantic duplicates (threshold ≥ 0.85)
- Stores each fact individually with `infer=False, graph=False` (no LLM re-extraction)
- Graph extraction runs in **background thread**, chunked for long texts (~2000 chars/chunk)
- Returns detailed status: new / updated / duplicate

### Fact Extraction Improvements
- Splits on both newlines and sentence boundaries (handles bullets, headers, numbered lists)
- Protects file extensions (`.php`, `.js`), version numbers (`v3.3`, `1.17.0`), file paths
- Minimum fact length: 35 chars (filters headers and label-only lines)
- Strips bullet markers (`*`, `-`, numbered items)
- Rejects section headers (short title-cased lines without content)

**Result**: 93% of facts include project context (was 0% before context injection).
171 well-formed facts from a 17KB document (was 38 multi-paragraph blobs or 233 fragments).

### Async Chunked Graph Extraction
Long texts are split into ~2000 char chunks for graph extraction. Each chunk gets a
separate LLM call in a background thread. The API responds immediately after vector
storage (~0.5s). Graph populates within 10-30 seconds asynchronously for large documents.

**Result**: 32 nodes with diverse types from a 17KB document (was 0 nodes when sent as
one giant text — LLM couldn't handle the full context).

## 4. Custom Prompts for Small Models (`openmemory/api/custom_update_prompt.py`)

Default mem0 prompts are designed for GPT-4 and are too verbose for 4B models.

Three optimized prompts:
- **Fact extraction**: Forces JSON-only output, context-preserving examples (each fact must
  include its subject), today's date for context
- **Update memory**: Clear decision rules (ADD/UPDATE/DELETE/NONE) with bias toward
  preserving information rather than over-deduplicating
- **Graph relationships**: 11 entity types with hierarchy rules, concrete example output,
  relationship verbs (develops, features, has_phase, has_target, built_with, etc.)

## 5. Core mem0 Library Changes

### `mem0/memory/main.py`
- Added `graph: bool = True` parameter to `Memory.add()`
- When `graph=False`, skips graph extraction in the ThreadPoolExecutor
- Allows per-fact vector storage without redundant per-fact graph calls

### `mem0/memory/graph_memory.py`
- **Entity list bug fix**: When `custom_prompt` was set, the user message omitted the
  entity list, causing relationship extraction to fail. Now always includes entity list.
- Added tech-specific entity type rules to the entity extraction system prompt
- **Self-referential edge guard**: In `_add_entities()`, compares resolved Neo4j element IDs
  before creating edges. Prevents self-loops when different entity names (e.g., "water" and
  "water element") resolve to the same physical node via embedding similarity.

## 6. MCP Server Enhancements (`openmemory/api/app/mcp_server.py`)

### Tools (7 total)

| Tool | Description |
|------|-------------|
| `add_memories` | Smart add with dedup (via `enhanced_memory_manager`) |
| `search_memory` | Hybrid search: vector + graph + temporal (limit: 10, offset pagination) |
| `list_memories` | List all memories with permission filtering |
| `delete_memories` | Smart delete: returns deleted content + related memories for cascade |
| `delete_all_memories` | Bulk deletion with state management |
| `handle_conversation` | Process user message + LLM response, extract memorable content |
| `get_related_memories` | Relationship traversal, grouped by source (vector/graph/temporal) |

### Permission Filter Fix
Graph and temporal results have generated UUIDs (not in SQLite). The permission filter
checked `result.id in accessible_memory_ids`, silently dropping all non-vector results.
Fixed: graph/temporal results bypass permission check (they have no SQLite entry).

### Search Pagination
`search_memory` MCP tool and `POST /api/v1/memories/search` REST endpoint support `offset`
parameter for pagination. Fetches `limit + offset` results from hybrid search, then slices.
Response includes `total_available`, `offset`, `has_more` for cursor-style paging.

### Improved Tool Descriptions
Tool descriptions rewritten to guide LLMs on correct input format:
- `add_memories`: "one fact per line, self-contained with subject"
- `search_memory`: explains offset pagination and query angle tips
- `handle_conversation`: clarifies use case vs add_memories

### Conversation Memory Filter Fix
`_extract_memorable_content()` in `enhanced_memory.py` had an aggressive keyword filter
that required LLM facts to contain words like "recommend" or "suggest" — dropped 95%+ of
useful content. Replaced with a filler-phrase exclusion filter (skips "sure", "let me",
"here is", etc.) and minimum length threshold (40 chars).

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

## 10. MEMORY_MODE: Advanced vs Simple

**Env var**: `MEMORY_MODE` (default: `advanced`)

Controls whether the REST API and MCP server use enhanced features or upstream-like behavior.

| Feature | `advanced` | `simple` |
|---------|-----------|----------|
| **Search** | Hybrid: vector + graph + temporal | SQLite `ILIKE` text match |
| **Add** | Smart add: fact splitting, dedup (≥0.85), async graph | Raw `memory_client.add()` |
| **Delete** | Returns deleted content + related memories | Returns count only |

### `POST /api/v1/memories/search` (new endpoint)
Hybrid search endpoint for AI consumers. Keeps `GET /` untouched for UI pagination.

```json
// Request
{"query": "steven", "user_id": "steven", "limit": 10}

// Response (advanced)
{
  "query": "steven",
  "results": [
    {"id": "...", "memory": "...", "score": 0.85, "source": "vector", "created_at": "..."}
  ],
  "total": 8,
  "sources_used": ["vector", "graph", "temporal"]
}
```

In `simple` mode, falls back to SQLite `ILIKE` search with score 1.0 and source "text_match".

### Smart Add Response (advanced mode)
`POST /api/v1/memories/` returns extra fields:
- `skipped_duplicates`: count of facts rejected by dedup
- `related_memories`: top 5 similar existing memories with scores
- `summary`: human-readable status message

### Smart Delete Response (advanced mode)
`DELETE /api/v1/memories/` returns:
- `deleted`: list of `{id, memory}` showing what was removed
- `related_memories`: similar memories with scores/sources for AI cascade decisions
- `message`: summary with counts

MCP server tools (`delete_memories`, `search_memory`, `add_memories`) respect the same
`MEMORY_MODE` setting and return equivalent rich responses.

## 11. Memory Router (`openmemory/api/app/routers/memories.py`)

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

## 12. Database (`openmemory/api/app/database.py`)

SQLite persistence fix:
```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/openmemory.db")
os.makedirs("db", exist_ok=True)
```
Ensures `./db/` directory exists for Docker volume mounting (`openmemory-db` volume).

## 13. Docker Compose Override (`openmemory/docker-compose.override.yml`)

- **Neo4j 5.26.4**: Graph database with APOC plugin, ports 8474/8687, healthcheck
- **Qdrant init**: Pre-creates `openmemory` collection (1024-dim cosine)
- `extra_hosts: host.docker.internal:host-gateway` for Ollama access from containers
- Shared `openmemory_network` bridge network
- Service dependency ordering with healthchecks

## 14. Qdrant Init Script (`openmemory/init-qdrant.sh`)

Waits for Qdrant readiness, then creates `openmemory` collection with 1024-dim Cosine
vectors. Lets mem0 auto-create `mem0migrations` collection. Non-fatal if collection exists.

## 15. Requirements Changes

### `openmemory/api/requirements.txt`
- Added `langchain-neo4j` — Required for mem0's Neo4j graph store
- Added `rank-bm25` — BM25 text search for hybrid search

## 16. Test Suite (`test_memory_system.py`)

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
| `openmemory/api/app/utils/enhanced_memory.py` | ~620 | Hybrid search, smart add, context injection, chunked graph |
| `openmemory/api/fix_graph_entity_parsing.py` | ~225 | JSON-based graph extraction with improved prompt |
| `openmemory/api/custom_update_prompt.py` | ~150 | Context-preserving prompts for qwen3:4b |
| `openmemory/docker-compose.override.yml` | ~50 | Neo4j + networking |
| `openmemory/init-qdrant.sh` | ~20 | Collection initialization |
| `mcp-server/server.js` | ~200 | Standalone MCP server (stdio) |
| `mcp-server/SKILL.md` | ~50 | Claude Code skill definition |
| `test_memory_system.py` | ~300 | Comprehensive test suite |
| `TESTING_GUIDE.md` | ~180 | Testing checklist for extraction pipeline changes |

### Modified (vs upstream)

| File | Key Changes |
|------|-------------|
| `openmemory/api/app/utils/memory.py` | Ollama fix (~300 lines), config cascade, Docker host detection |
| `openmemory/api/app/utils/categorization.py` | OpenAI → Ollama with JSON fallback parsing |
| `openmemory/api/app/mcp_server.py` | Enhanced memory manager, 7 tools, permission filter fix, smart delete |
| `openmemory/api/app/routers/memories.py` | MEMORY_MODE, POST /search, smart add/delete, 7 graph endpoints |
| `openmemory/api/app/routers/config.py` | Env-based defaults, no auto-create DB rows |
| `openmemory/api/app/database.py` | SQLite path for Docker volume persistence |
| `openmemory/api/config.json` | Env var references for all providers |
| `mem0/memory/main.py` | `graph=False` parameter on `Memory.add()` |
| `mem0/memory/graph_memory.py` | Entity list bug fix, tech entity types |
| `openmemory/api/requirements.txt` | +langchain-neo4j, +rank-bm25 |
