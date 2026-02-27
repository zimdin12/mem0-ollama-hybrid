# Fork Changes: Ollama + Hybrid Memory (Local-Only Mode)

This document describes all changes made in this fork (`mem0-ollama-hybrid`) relative
to the upstream [mem0ai/mem0](https://github.com/mem0ai/mem0) repository.

**Goal**: Run OpenMemory fully locally using Ollama (no OpenAI dependency), with hybrid
memory mode enabled (vector store + graph database).

---

## 1. Ollama LLM Compatibility Fix (`openmemory/api/app/utils/memory.py`)

**Why**: mem0 expects LLM responses in specific formats (tool_calls dicts for graph
extraction, plain strings for fact extraction). Small local models like `qwen3:4b` return
responses in inconsistent formats — sometimes as plain text, sometimes as partial JSON,
sometimes with verbose parenthetical notes.

**What**: Added `_apply_essential_ollama_fix()` that monkey-patches
`OllamaLLM.generate_response()` to:
- Return **strings** for fact extraction (fixes `AttributeError: 'dict' has no 'strip'`)
- Return **dicts with tool_calls** for entity/relationship extraction
- Parse multiple qwen3 output formats into proper tool_calls:
  - `extract_entities([...])` function call syntax
  - Bare JSON arrays `[{"source":..., "relationship":...}]`
  - Structured text (Entity/Type blocks, bullet lists)
  - Mixed format arrays (strings and dicts)

## 2. Graph Entity Extraction Fix (`fix_graph_entity_parsing.py`)

**Why**: qwen3:4b adds parenthetical notes and verbose explanations to entity names
(e.g., `"Steven (source entity"` instead of `"Steven"`), breaking graph storage.

**What**: Monkey-patches `MemoryGraph._retrieve_nodes_from_data` and
`_establish_nodes_relations_from_data` with:
- Concise, directive prompts that force tool calling
- Regex cleaning for entity names/types (removes parentheses, arrows, quotes)
- Handles mixed formats (strings and dicts in entity arrays)
- Type guessing fallback for overly verbose entity_type fields

## 3. Custom Prompts for Small Models (`custom_update_prompt.py`)

**Why**: Default mem0 prompts are designed for GPT-4 and are too verbose/ambiguous for
4B-parameter models. Small models need explicit, structured instructions.

**What**: Three optimized prompts:
- **Fact extraction**: Forces JSON-only output with many examples of breaking prose
  into individual facts
- **Update memory**: Clear decision rules (ADD/UPDATE/DELETE/NONE) with bias toward
  preserving information rather than over-deduplicating
- **Graph relationships**: Comprehensive entity and relationship extraction with
  explicit format examples

## 4. Config Files Changed to Environment Variables

### `openmemory/api/config.json` & `default_config.json`

**Why**: Hardcoded OpenAI provider and API key references don't work for local Ollama.

**What**: All provider/model/URL fields use `env:VARIABLE_NAME` pattern:
- `LLM_PROVIDER`, `LLM_MODEL`, `OLLAMA_BASE_URL`
- `EMBEDDER_PROVIDER`, `EMBEDDER_MODEL`, `EMBEDDER_OLLAMA_BASE_URL`
- `VECTOR_STORE_PROVIDER`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

Also added **graph_store** section (neo4j) and **vector_store** section (qdrant) which
were absent from the upstream config.

### `openmemory/api/app/utils/memory.py` — Config loading

**Why**: Upstream only loads config from database. We need config.json as a reliable
fallback (especially for first startup before any UI config is saved).

**What**: Config loading order: `config.json` → database override → environment variable
parsing. Added integer conversion for `embedding_model_dims` and `port` fields.
Added `graph_store` loading support alongside existing llm/embedder/vector_store.

## 5. Categorization Changed to Ollama (`openmemory/api/app/utils/categorization.py`)

**Why**: Used hardcoded `OpenAI()` client and `gpt-4o-mini` with structured output
(`beta.chat.completions.parse`), which requires an OpenAI API key.

**What**:
- Uses `OpenAI(base_url=OLLAMA_BASE_URL/v1, api_key="ollama")` compatible client
- Regular `chat.completions.create` instead of structured output (Ollama doesn't support it)
- Manual JSON parsing with markdown code block extraction fallback
- Returns empty list on failure instead of raising (prevents crashes on parse errors)

## 6. Enhanced Memory Manager (`openmemory/api/app/utils/enhanced_memory.py`)

**Why**: Upstream MCP tools do basic vector-only search. With a graph database available,
we want hybrid search across all memory dimensions.

**What**: New `EnhancedMemoryManager` class providing:
- **Hybrid search**: Vector similarity + graph relationship + temporal recency search
- **Smart add**: Checks existing memories before adding, skips duplicates
- **Comprehensive conversation handling**: Extracts memorable content from both user
  messages and LLM responses

## 7. MCP Server Changes (`openmemory/api/app/mcp_server.py`)

**Why**: Upstream MCP tools use simple vector-only search and basic memory addition.
With hybrid mode, we want richer search and smarter addition.

**What**:
- `add_memories`: Uses `enhanced_memory_manager.smart_add_memory()` for deduplication
- `search_memory`: Uses `enhanced_memory_manager.hybrid_search()` across all dimensions
- Added `handle_conversation` tool: Processes both user message + LLM response
- Added `get_related_memories` tool: Relationship traversal via graph

## 8. Memory Router Changes (`openmemory/api/app/routers/memories.py`)

**Why**: Upstream router doesn't handle graph queries or long document chunking.

**What**:
- `CreateMemoryRequest` accepts `messages[]` in addition to `text` (for conversation format)
- Graph query helpers (`query_graph`, `get_memory_entities_from_graph`, etc.)
- Fallback chunking for long texts that mem0 returns empty results for
- Direct Qdrant vector storage as last-resort fallback

## 9. Docker Compose Override (`openmemory/docker-compose.override.yml`)

**Why**: Upstream doesn't include Neo4j or configure networking for Ollama access.

**What**:
- **Neo4j** container (5.26.4) with APOC plugin, healthcheck, mapped to ports 8474/8687
- **Qdrant** (`mem0_store`) with custom init script, healthcheck, curl install
- `openmemory-mcp` depends on neo4j healthy + mem0_store healthy
- `extra_hosts: host.docker.internal:host-gateway` for Ollama access from containers
- `openmemory-ui` build args for custom API URL and user ID
- Shared `openmemory_network` bridge network

## 10. Qdrant Init Script (`openmemory/init-qdrant.sh`)

**Why**: Qdrant needs the `openmemory` collection pre-created with correct dimensions
(1024 for `qwen3-embedding:0.6b`). Without it, mem0 fails on first write.

**What**: Shell script that waits for Qdrant readiness, then creates `openmemory`
collection with 1024-dim Cosine vectors. Lets mem0 auto-create `mem0migrations`.

## 11. Requirements Changes

### `openmemory/api/requirements.txt`
- Added `langchain-neo4j` — required for mem0's Neo4j graph store integration
- Added `rank-bm25` — used for BM25 text search in hybrid search

### `server/requirements.txt`
- Added `psycopg2` — PostgreSQL adapter (needed alongside psycopg3)

## 12. Server Docker Compose (`server/docker-compose.yaml`)

**What**: Changed default credentials:
- PostgreSQL: `postgres/postgres` → `mem0/openmemory`
- Neo4j: `neo4j/mem0graph` → `neo4j/openmemory`

---

## Environment Variables (`.env`)

Required environment variables for local Ollama operation:

```env
# LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://192.168.100.10:11434  # or host.docker.internal:11434
LLM_MODEL=qwen3:4b-instruct

# Embeddings
EMBEDDER_PROVIDER=ollama
EMBEDDER_MODEL=qwen3-embedding:0.6b
EMBEDDER_OLLAMA_BASE_URL=http://192.168.100.10:11434

# Graph (Neo4j)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=openmemory

# Vector (Qdrant)
VECTOR_STORE_PROVIDER=qdrant
QDRANT_HOST=mem0_store
QDRANT_PORT=6333
QDRANT_COLLECTION=openmemory
QDRANT_EMBEDDING_DIMS=1024
```

## Files Added (Not in Upstream)

| File | Purpose |
|------|---------|
| `openmemory/docker-compose.override.yml` | Neo4j + Qdrant init + networking |
| `openmemory/init-qdrant.sh` | Pre-create Qdrant collection |
| `openmemory/api/app/utils/enhanced_memory.py` | Hybrid search + smart add |
| `fix_graph_entity_parsing.py` | Qwen3 graph extraction fix |
| `custom_update_prompt.py` | Optimized prompts for small models |

## Files Modified (vs Upstream)

| File | Key Change |
|------|------------|
| `openmemory/api/app/utils/memory.py` | Ollama fix, config.json loading, graph_store support |
| `openmemory/api/app/utils/categorization.py` | OpenAI → Ollama |
| `openmemory/api/app/mcp_server.py` | Enhanced memory manager integration |
| `openmemory/api/app/routers/memories.py` | Graph queries, chunking, messages[] support |
| `openmemory/api/config.json` | Env vars, graph_store, vector_store |
| `openmemory/api/default_config.json` | Env vars, graph_store, vector_store |
| `openmemory/api/requirements.txt` | +langchain-neo4j, +rank-bm25 |
| `server/docker-compose.yaml` | Credential changes |
| `openmemory/README.md` | Fork documentation |
