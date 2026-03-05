# mem0 — Local Hybrid Memory (Ollama Fork)

Fork of [mem0ai/mem0](https://github.com/mem0ai/mem0) that runs **100% locally** using Ollama. No OpenAI, no cloud APIs, no API keys needed.

Provides a persistent hybrid memory system (vector + graph) for AI assistants, agents, and coding tools.

## What It Does

When you store a memory like *"Steven prefers TypeScript and uses an RTX 4090"*, mem0:

1. **Extracts facts** → individual memory entries in the vector store
2. **Extracts entities** → `Steven`, `TypeScript`, `RTX 4090` as graph nodes
3. **Extracts relationships** → `Steven --prefers--> TypeScript`, `Steven --uses--> RTX 4090`
4. **Deduplicates** → if you already stored similar info, it updates instead of duplicating

Searching for *"what GPU does Steven use"* returns results from both the vector store (semantic similarity) and the knowledge graph (entity relationships).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Clients                         │
│  OpenClaw · Claude Code · OpenCode · Any MCP     │
└──────────────┬──────────────────┬───────────────┘
               │  REST API        │  MCP (SSE)
        ┌──────▼──────────────────▼───────┐
        │      OpenMemory API (:8765)     │
        │         (FastAPI + mem0)        │
        └──┬──────────┬──────────┬────────┘
           │          │          │
     ┌─────▼───┐ ┌───▼────┐ ┌──▼──────┐
     │ Qdrant  │ │ Neo4j  │ │ Ollama  │
     │ :6333   │ │ :8687  │ │ :11434  │
     │ vectors │ │ graph  │ │ LLM+Emb │
     └─────────┘ └────────┘ └─────────┘
```

| Component | Purpose | Port |
|-----------|---------|------|
| **OpenMemory API** | REST + MCP server, orchestrates mem0 | 8765 |
| **OpenMemory UI** | Web dashboard for browsing memories | 3100 |
| **Qdrant** | Vector store (semantic search) | 6333 |
| **Neo4j** | Graph store (entity relationships) | 8474 / 8687 |
| **Ollama** | LLM for extraction, embeddings | 11434 |
| **SQLite** | Metadata, audit trail | file |

## Models Used (via Ollama)

| Role | Model | Size | Notes |
|------|-------|------|-------|
| Fact & entity extraction | `qwen3:4b-instruct-2507-q4_K_M` | ~2.5 GB | Must be `instruct` variant (thinking mode breaks tool calls) |
| Embeddings | `qwen3-embedding:0.6b` | ~0.5 GB | 1024 dimensions, cosine similarity |

Both run on GPU via Ollama. Total VRAM: ~3 GB.

## Deployment

### As part of OpenClaw (recommended)

This repo is used as a git submodule in [openclaw-adv-mem-local](https://github.com/zimdin12/openclaw-adv-mem-local). The parent repo's `docker-compose.yml` defines all services (Qdrant, Neo4j, Ollama, API, UI) with the correct networking and env vars. **Do not use the `openmemory/` compose files** when running inside the parent stack.

### Standalone

```bash
# 1. Clone
git clone https://github.com/zimdin12/mem0-ollama-hybrid.git
cd mem0-ollama-hybrid

# 2. Pull Ollama models (requires Ollama running on the host)
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-embedding:0.6b

# 3. Copy env files
cd openmemory
cp api/.env.example api/.env    # edit if Ollama is not on the default host
cp ui/.env.example ui/.env

# 4. Start services
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# 5. Verify
curl http://localhost:8765/docs    # API docs
open http://localhost:3000         # Memory UI
```

### Environment Variables

There are **3 separate .env files** for different components:

| File | What it configures | When to edit |
|------|-------------------|--------------|
| `openmemory/api/.env.example` | **API container** — LLM model, embedder, Qdrant, Neo4j, Ollama URL | Standalone deployment (not used when parent compose sets env vars) |
| `openmemory/ui/.env.example` | **UI dashboard** — API URL and default user ID | Only if UI runs on a different host |
| `mcp-server/.env.example` | **Host-side MCP server** (stdio transport) — API URL, user ID, app name | Only if using stdio MCP instead of SSE |

**When running inside a parent stack** (like OpenClaw), the parent's `docker-compose.yml` sets all env vars directly in the `environment:` section — the API `.env` file is NOT used. Edit `docker-compose.yml` instead.

**When running standalone**, copy `openmemory/api/.env.example` to `.env` and edit it.

#### MEMORY_MODE

Controls whether the API uses enhanced features or basic mem0 operations:

```env
MEMORY_MODE=advanced   # (default) hybrid search, smart add with dedup, smart delete with related discovery
MEMORY_MODE=simple     # basic mem0: SQLite text search, raw add, simple delete count
```

Set in `docker-compose.yml` (parent stack) or `openmemory/api/.env` (standalone).

#### API Environment Variables (the important ones)

```env
# === Ollama — where is it running? ===
# Same Docker network:              http://ollama:11434
# Host machine (Docker Desktop):    http://host.docker.internal:11434
# Remote machine:                   http://192.168.x.x:11434
OLLAMA_BASE_URL=http://ollama:11434

# === LLM (fact & entity extraction) ===
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M   # Any Ollama chat model works

# === Embeddings ===
EMBEDDER_PROVIDER=ollama
EMBEDDER_MODEL=qwen3-embedding:0.6b         # Any Ollama embedding model works
EMBEDDER_OLLAMA_BASE_URL=http://ollama:11434

# === Vector Store (Qdrant) ===
VECTOR_STORE_PROVIDER=qdrant
QDRANT_HOST=mem0_store
QDRANT_PORT=6333
QDRANT_COLLECTION=openmemory
QDRANT_EMBEDDING_DIMS=1024                   # Must match embedding model output dims!

# === Graph Store (Neo4j) ===
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=openmemory
```

### Using Different Models

Any Ollama model works as a drop-in replacement. The fork's compatibility layers handle
all Ollama output formats automatically.

**To change the extraction LLM** (affects fact/entity quality):
```env
LLM_MODEL=llama3.1:8b          # or glm4:9b, magistral:24b, etc.
```
Bigger models = better entity extraction quality but slower. The 4B sweet spot balances
speed (~3s graph extraction) with quality (79+ entities from test data).

**To change the embedding model** (affects search quality):
```env
EMBEDDER_MODEL=nomic-embed-text       # 768 dims
QDRANT_EMBEDDING_DIMS=768             # MUST match!
```
**Warning**: Changing embedding model or dimensions requires:
1. Wiping the Qdrant collection (old vectors are incompatible)
2. Re-inserting all memories

Common embedding models and their dimensions:

| Model | Dims | Size | Notes |
|-------|------|------|-------|
| `qwen3-embedding:0.6b` (default) | 1024 | 639 MB | Best speed/quality for short facts |
| `qwen3-embedding:4b` | 2560 (or 1024 via MRL) | 2.5 GB | Better on MTEB benchmarks but 5.7x slower |
| `nomic-embed-text` | 768 | 274 MB | Older, lower quality |
| `snowflake-arctic-embed2:568m` | 1024 | 1.2 GB | Good multilingual |

### Repository Structure

```
mem0-fork/
├── openmemory/                    # Docker services (API + UI)
│   ├── api/                       # FastAPI backend (Python)
│   │   ├── app/routers/           # REST endpoints
│   │   ├── app/utils/             # Memory client, Ollama fixes
│   │   ├── config.json            # mem0 config (env var references)
│   │   ├── Dockerfile
│   │   └── .env.example
│   ├── ui/                        # Next.js dashboard
│   │   ├── Dockerfile
│   │   └── .env.example
│   ├── docker-compose.yml         # Base services (Qdrant, API, UI)
│   ├── docker-compose.override.yml # Neo4j, networking, health checks
│   └── init-qdrant.sh             # Collection init (1024d cosine)
├── mcp-server/                    # Host-side MCP server (NOT containerized)
│   ├── server.js                  # Node.js stdio transport, 4 tools
│   ├── SKILL.md                   # Claude Code skill (auto-recall/capture)
│   ├── package.json
│   └── .env.example
├── mem0/                          # Core mem0 library (Python, from upstream)
└── README.md
```

**Important:** `mcp-server/` runs on the **host machine** (not in Docker). It's a Node.js process that Claude Code or OpenCode spawn via stdio. It talks to the OpenMemory API over HTTP.

## Connecting Clients

There are two ways to connect MCP clients to OpenMemory:

| Method | Transport | Requires | Best for |
|--------|-----------|----------|----------|
| **SSE** (recommended) | HTTP to API container | Nothing extra | Any MCP client — no Node.js needed |
| **Stdio** | Node.js process on host | `npm install` in `mcp-server/` | Claude Code / OpenCode if SSE isn't supported |

### SSE Endpoint (recommended)

The OpenMemory API has a **built-in MCP server** at:

```
http://<host>:8765/mcp/<client_name>/sse/<user_id>
```

Replace `<host>` with your machine's IP or `localhost`, `<client_name>` with any label, and `<user_id>` with your user ID.

**Claude Code:**

```bash
claude mcp add openmemory --transport sse http://localhost:8765/mcp/claude-code/sse/steven
```

Or add to `~/.claude.json` (global) or `.mcp.json` (per-project):

```json
{
  "mcpServers": {
    "openmemory": {
      "type": "sse",
      "url": "http://localhost:8765/mcp/claude-code/sse/steven"
    }
  }
}
```

**OpenCode:**

```json
{
  "openmemory": {
    "type": "sse",
    "url": "http://localhost:8765/mcp/opencode/sse/steven"
  }
}
```

**Any MCP client (generic SSE config):**

```json
{
  "type": "sse",
  "url": "http://192.168.x.x:8765/mcp/my-client/sse/steven"
}
```

### Stdio (Node.js MCP server)

Alternative if your client doesn't support SSE. Runs `mcp-server/server.js` on the host machine as a subprocess.

```bash
cd mcp-server && npm install
```

**Claude Code:**

```bash
claude mcp add openmemory -- node /path/to/mcp-server/server.js
```

Or in config:

```json
{
  "mcpServers": {
    "openmemory": {
      "type": "stdio",
      "command": "node",
      "args": ["/path/to/mcp-server/server.js"],
      "env": {
        "MEMORY_API_URL": "http://localhost:8765",
        "MEMORY_USER_ID": "steven"
      }
    }
  }
}
```

See `mcp-server/.env.example` for all env vars.

**OpenCode:**

```json
{
  "openmemory": {
    "type": "stdio",
    "command": "node",
    "args": ["/path/to/mcp-server/server.js"],
    "env": {
      "MEMORY_API_URL": "http://localhost:8765",
      "MEMORY_USER_ID": "steven"
    }
  }
}
```

### Claude Code Skill (optional, works with either transport)

The skill tells Claude *when* to use memory tools (auto-recall, auto-capture). It complements the MCP server — doesn't replace it.

Copy into Claude Code's skill discovery directory:

```bash
# Project-level (this repo only)
mkdir -p .claude/skills/openmemory
cp mcp-server/SKILL.md .claude/skills/openmemory/SKILL.md

# Or personal (all projects)
mkdir -p ~/.claude/skills/openmemory
cp mcp-server/SKILL.md ~/.claude/skills/openmemory/SKILL.md
```

Claude Code auto-discovers skills from `.claude/skills/` — no CLI command needed.

### OpenClaw

Uses the bundled plugin at `config/extensions/openmemory/`. Config in `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "openmemory": {
        "enabled": true,
        "config": {
          "apiUrl": "http://openmemory-mcp:8765",
          "userId": "steven"
        }
      }
    }
  }
}
```

### REST API

```bash
# Hybrid search (advanced mode: vector + graph + temporal)
curl -X POST http://localhost:8765/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what GPU does steven use", "user_id": "steven", "limit": 10}'

# Store a memory (advanced mode: smart add with dedup)
# Returns: added facts, skipped duplicates, related existing memories
curl -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Steven uses an RTX 4090", "user_id": "steven"}'

# Delete (advanced mode: returns deleted content + related memories)
curl -X DELETE http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"memory_ids": ["<uuid>"], "user_id": "steven"}'

# List memories (UI pagination, not hybrid - always available)
curl "http://localhost:8765/api/v1/memories/?user_id=steven&size=10"

# Graph context
curl "http://localhost:8765/api/v1/memories/graph/context/Steven?user_id=steven"
```

## How This Fork Differs from Upstream

Upstream mem0 defaults to OpenAI for everything. This fork replaces all cloud dependencies with local Ollama equivalents.

### Key Changes

| Area | Upstream | This Fork |
|------|----------|-----------|
| LLM | OpenAI GPT-4 | Ollama (qwen3:4b-instruct) |
| Embeddings | OpenAI ada-002 | Ollama (qwen3-embedding:0.6b, 1024d) |
| Config | Hardcoded OpenAI keys | Environment variables (cascade: config.json → env) |
| Categorization | OpenAI structured output | Ollama + manual JSON parsing |
| Graph extraction | GPT-4 tool calling (3-4 LLM calls) | Single Ollama JSON mode call (`format:'json'`) |
| Fact extraction | Naive sentence splitting | Protected splitting (preserves file extensions, versions, paths) |
| Prompts | Verbose (for GPT-4) | Concise, directive (for 4B models) |
| Memory search | Vector only | Hybrid: vector + graph + temporal (interleaved 60/30/10) |
| Memory addition | Synchronous, per-fact graph | Async graph extraction in background thread (5x faster) |
| Memory deletion | Returns count only | Smart delete: returns deleted content + related memories |
| Deduplication | Basic | Three-layer: cosine ≥ 0.95, vector ≥ 0.85, infer=False |
| REST API | Basic endpoints | `MEMORY_MODE` switch: `advanced` uses hybrid search/smart add/delete |
| MCP tools | Basic (4 tools) | Enhanced (7 tools: hybrid search, smart add, graph traversal) |
| MCP permissions | Filters out graph results | Correctly passes graph/temporal results through |
| `Memory.add()` | No control over graph | Added `graph=False` parameter for per-fact control |

### Files Added

| File | Purpose |
|------|---------|
| `openmemory/api/app/utils/enhanced_memory.py` | Hybrid search, smart dedup, async graph extraction |
| `openmemory/api/fix_graph_entity_parsing.py` | JSON-based graph extraction (replaces tool calling) |
| `openmemory/api/custom_update_prompt.py` | Optimized prompts for qwen3:4b |
| `openmemory/docker-compose.override.yml` | Neo4j + Qdrant init + networking |
| `openmemory/init-qdrant.sh` | Pre-create Qdrant collection (1024d cosine) |
| `mcp-server/server.js` | Standalone MCP server (4 tools, stdio transport) |
| `mcp-server/SKILL.md` | Claude Code skill for auto-recall/capture behavior |
| `test_memory_system.py` | 5-phase test suite (insert, validate, graph, search, dedup) |

### Files Modified

| File | Change |
|------|--------|
| `openmemory/api/app/utils/memory.py` | Ollama compatibility, env-based config, graph store |
| `openmemory/api/app/utils/categorization.py` | OpenAI → Ollama |
| `openmemory/api/app/mcp_server.py` | Enhanced memory manager, graph/temporal filter fix |
| `openmemory/api/app/routers/memories.py` | Graph endpoints, DELETE sync, no phantoms |
| `openmemory/api/app/routers/config.py` | Env-based defaults, no auto-create DB rows |
| `openmemory/api/app/database.py` | SQLite path fix for Docker volume persistence |
| `openmemory/api/config.json` | Env var references for all providers |
| `mem0/memory/main.py` | Added `graph=False` parameter to `Memory.add()` |
| `mem0/memory/graph_memory.py` | Entity list bug fix, tech entity types |

For the full technical changelog, see [FORK_CHANGES.md](FORK_CHANGES.md).

## Performance

Benchmarked on RTX 4090 with qwen3:4b + qwen3-embedding:0.6b:

| Operation | Latency | Notes |
|-----------|---------|-------|
| Hybrid search | 90-135ms | Embedding + Qdrant + Neo4j + dedup |
| Add (short text) | ~0.4s | Vector stored immediately |
| Add (medium text) | ~0.5s | Multiple facts extracted |
| Add (long text) | ~0.9s | Many facts, dedup checks |
| Graph population | 3-5s | Runs async after API response |

Embedding model comparison (qwen3-embedding):
- **0.6b** (current): 98ms/embed, optimal for short facts
- **4b**: 562ms/embed (5.7x slower), no quality gain for memory-length texts

## Data Persistence

| Data | Storage | Docker Volume |
|------|---------|---------------|
| Vector embeddings | Qdrant | `qdrant-data` |
| Knowledge graph | Neo4j | `neo4j-data` |
| Memory metadata | SQLite | `openmemory-db` |

All data survives container restarts and redeployments.

## Roadmap

Planned features (not yet implemented):

- **JSON Export/Import** — `GET /api/v1/memories/export` dumps all memories as portable JSON
  (text + timestamps + categories). `POST /api/v1/memories/import` re-ingests via `smart_add`
  pipeline (auto-embeds, deduplicates, rebuilds graph). Enables model migration: switch
  embedding model, re-import, and all vectors + graph nodes are regenerated.
- **MCP export/import tools** — `export_memories` and `import_memories` for Claude Code / MCP clients
- **Async import with progress** — large imports (hundreds of memories) with status tracking

## License

Apache 2.0 — same as upstream. See [LICENSE](LICENSE).

Based on [mem0ai/mem0](https://github.com/mem0ai/mem0) by the Mem0 team.
