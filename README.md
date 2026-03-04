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

All env vars are documented in `.env.example` files:

| File | Purpose |
|------|---------|
| `openmemory/api/.env.example` | API server — LLM, embedder, Qdrant, Neo4j config |
| `openmemory/ui/.env.example` | UI dashboard — API URL and user ID |
| `mcp-server/.env.example` | MCP server (host-side) — API URL and user ID |

Key variables for the API container:

```env
# LLM (fact & entity extraction)
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
OLLAMA_BASE_URL=http://ollama:11434        # Docker service name
# or http://host.docker.internal:11434     # host Ollama

# Embeddings
EMBEDDER_PROVIDER=ollama
EMBEDDER_MODEL=qwen3-embedding:0.6b
EMBEDDER_OLLAMA_BASE_URL=http://ollama:11434

# Qdrant
VECTOR_STORE_PROVIDER=qdrant
QDRANT_HOST=mem0_store
QDRANT_PORT=6333
QDRANT_COLLECTION=openmemory
QDRANT_EMBEDDING_DIMS=1024

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=openmemory
```

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

### Claude Code

Two parts: **MCP server** (provides 4 memory tools) and **Skill** (tells Claude when/how to use them).

```bash
cd mcp-server && npm install
```

**1. Register MCP server** (one of these methods):

CLI:
```bash
claude mcp add openmemory -- node /path/to/mcp-server/server.js
```

Or add to `~/.claude.json` (global) or `.mcp.json` (per-project):
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

**2. Install skill** (one of these methods):

CLI:
```bash
claude skill add /path/to/mcp-server/SKILL.md
```

Or copy `SKILL.md` into your project's `.claude/skills/` directory.

The skill adds auto-recall at conversation start, auto-capture of durable facts, and guidelines for writing good memories. It requires the MCP server to be registered (the skill references the MCP tools).

### OpenCode (MCP only)

Add to your OpenCode MCP config:

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

### SSE Endpoint (any MCP client)

```
http://localhost:8765/mcp/{client_name}/sse/{user_id}
```

Example: `http://localhost:8765/mcp/opencode/sse/steven`

### REST API

```bash
# Search memories
curl "http://localhost:8765/api/v1/memories/?user_id=steven&search_query=gpu&size=5"

# Store a memory
curl -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Steven uses an RTX 4090", "user_id": "steven"}'

# Graph context
curl "http://localhost:8765/api/v1/memories/graph/context/Steven?user_id=steven"
```

## How This Fork Differs from Upstream

Upstream mem0 defaults to OpenAI for everything. This fork replaces all cloud dependencies with local Ollama equivalents.

### Key Changes

| Area | Upstream | This Fork |
|------|----------|-----------|
| LLM | OpenAI GPT-4 | Ollama (qwen3:4b-instruct) |
| Embeddings | OpenAI ada-002 | Ollama (qwen3-embedding:0.6b) |
| Config | Hardcoded OpenAI keys | Environment variables |
| Categorization | OpenAI structured output | Ollama + manual JSON parsing |
| Graph extraction | Assumes GPT-4 tool calling | Monkey-patched parser for qwen3 output formats |
| Prompts | Verbose (for GPT-4) | Concise, directive (for 4B models) |
| Memory search | Vector only | Hybrid: vector + graph + temporal |
| Config source | Database only | config.json → DB → env vars (cascade) |
| MCP tools | Basic | Enhanced (hybrid search, smart add, graph traversal) |

### Files Added

| File | Purpose |
|------|---------|
| `openmemory/docker-compose.override.yml` | Neo4j + Qdrant init + networking |
| `openmemory/init-qdrant.sh` | Pre-create Qdrant collection (1024d cosine) |
| `openmemory/api/app/utils/enhanced_memory.py` | Hybrid search + smart dedup |
| `fix_graph_entity_parsing.py` | Qwen3 graph extraction parser |
| `custom_update_prompt.py` | Optimized prompts for small models |
| `mcp-server/server.js` | Standalone MCP server (4 tools, stdio transport) |
| `mcp-server/SKILL.md` | Claude Code skill for auto-recall/capture behavior |

### Files Modified

| File | Change |
|------|--------|
| `openmemory/api/app/utils/memory.py` | Ollama compatibility, env-based config, graph store |
| `openmemory/api/app/utils/categorization.py` | OpenAI → Ollama |
| `openmemory/api/app/mcp_server.py` | Enhanced memory manager |
| `openmemory/api/app/routers/memories.py` | Graph queries, DELETE sync, no phantoms |
| `openmemory/api/app/routers/config.py` | Env-based defaults, no auto-create DB rows |
| `openmemory/api/app/database.py` | SQLite path fix for Docker volume persistence |
| `openmemory/api/config.json` | Env var references for all providers |

For the full technical changelog, see [FORK_CHANGES.md](FORK_CHANGES.md).

## Data Persistence

| Data | Storage | Docker Volume |
|------|---------|---------------|
| Vector embeddings | Qdrant | `qdrant-data` |
| Knowledge graph | Neo4j | `neo4j-data` |
| Memory metadata | SQLite | `openmemory-db` |

All data survives container restarts and redeployments.

## License

Apache 2.0 — same as upstream. See [LICENSE](LICENSE).

Based on [mem0ai/mem0](https://github.com/mem0ai/mem0) by the Mem0 team.
