# OpenMemory Fork — LLM Quick Reference

This file is for LLMs (Claude Code, OpenClaw, Open WebUI agents) that need to understand the memory system quickly.

## What This Is

A local-only fork of [mem0](https://github.com/mem0ai/mem0) that runs entirely on Ollama. Provides hybrid memory (vector + graph + temporal) for AI assistants.

**Branch: `main` (v1)** — 7 specific MCP tools for programmatic memory access.

## API Endpoints

```
POST /api/v1/memories/         — add memories (text, user_id, app_id)
POST /api/v1/memories/search   — hybrid search (query, user_id, limit, offset)
DELETE /api/v1/memories/       — batch delete (memory_ids, user_id)
GET  /api/v1/memories/         — list all (user_id, app_id)
POST /api/v1/memories/conversation — process conversation turn
GET  /api/v1/config/mem0       — current config (llm, embedder, stores)
PUT  /api/v1/config/mem0/llm   — change LLM model at runtime
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `add_memories` | Store facts (one per line, self-contained with subject) |
| `search_memory` | Hybrid search with offset pagination |
| `list_memories` | List all stored memories |
| `delete_memories` | Delete by content description |
| `delete_all_memories` | Clear everything |
| `conversation_memory` | Process conversation turns (regex + LLM review) |
| `get_related_memories` | Graph relationship traversal |

## Storage Architecture

```
Qdrant (port 6333)  — vector store, cosine similarity, 1024 dims
Neo4j (port 8687)   — graph store, entity relationships
SQLite              — metadata, users, apps, audit log
```

## Key Behaviors

- **Dedup**: New facts checked against existing (cosine >= 0.85 = update, >= 0.95 = skip)
- **Graph**: Async extraction after vector storage (3-5s background)
- **Search**: Interleaved results from vector (60%), graph (30%), temporal (10%)
- **Short input bypass**: Inputs <100 chars skip extraction pipeline
- **Noise filter**: LLM-based classification rejects greetings/filler

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `qwen3:4b-instruct-2507-q4_K_M` | Extraction model |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama API URL |
| `EMBEDDER_MODEL` | `qwen3-embedding:0.6b` | Embedding model |
| `NEO4J_URL` | `bolt://neo4j:7687` | Neo4j connection |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| `LLM_FACT_REVIEW` | `true` | Enable LLM review in conversation_memory |

## For Detailed Docs

- `README.md` — full deployment guide, all env vars, client setup
- `FORK_CHANGES.md` — technical changelog vs upstream
- `TESTING_GUIDE.md` — test checklist for extraction pipeline changes
- `CLAUDE.md` — Claude Code specific setup
- `SYSTEM_PROMPT.md` — Open WebUI / LM Studio system prompt template
- `mcp-server/SKILL.md` — Claude Code skill (auto-recall behavior)
