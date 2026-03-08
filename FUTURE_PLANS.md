# Future Plans: Memory Agent (v2)

## Implementation Status

Branch: `memory_agent`

- [x] Phase 1: Brain tools (8 DB primitives + audit table)
- [x] Phase 2: Agent loop (JSON-mode, max_steps, retry logic)
- [x] Phase 3: REST API (single endpoint)
- [x] Phase 4: MCP tool (`memory_agent`)
- [x] Phase 5: Tests (22 tests, 3 suites)
- [x] Client configs updated (OpenClaw plugin, Claude Code skill, OpenWebUI prompt)
- [x] Production hardening (model quality tuning, edge case handling)
- [x] Performance benchmarking vs v1

## Vision

**v1** (main branch): The main LLM (Claude, Open WebUI) calls simple tools (`search_memory`,
`add_memories`). Responsibility is split — the main LLM decides when/how to use memory,
the memory system handles storage/retrieval. Fast, reliable, production-ready.

**v2** (memory_agent branch): The memory system becomes a full autonomous agent. The main LLM
just talks to it in natural language. The Memory Agent handles ALL complexity — deciding what
to search, how to store, when to deduplicate, how to synthesize answers. It's agent-to-agent
communication.

```
v1: Main LLM  ─── search_memory("GPU") ───>  code does hybrid search  ──> results
    Main LLM  ─── add_memories("fact")  ───>  code does extraction    ──> stored

v2: Main LLM  ─── "What GPU does Steven use?" ───>  Memory Agent (qwen3.5:4b)
                                                        ├── vector_search("steven GPU")
                                                        ├── graph_query(MATCH steven...)
                                                        └── synthesized answer back
```

## Architecture (v2)

### Design Principles

1. **Single endpoint** — one REST endpoint (`POST /api/v1/brain`), one MCP tool (`memory_agent`). The agent determines intent from the natural language request.
2. **Single model** — uses `LLM_MODEL` (same as extraction pipeline). One model for everything. No separate brain model.
3. **Full autonomy** — the agent has access to all 8 tools (search, store, delete, graph read/write, SQL read/write, embed). It decides the strategy.
4. **v1 coexists** — all 7 v1 MCP tools remain available for fast/simple operations.

### Tool Architecture

```
Memory Agent (LLM reasoning loop)
    │
    ├── vector_search(query, limit, user_id)     → Qdrant
    ├── vector_store(text, user_id)               → Qdrant + async graph
    ├── vector_delete(ids)                        → Qdrant + SQLite
    ├── graph_query(cypher, user_id)              → Neo4j (read-only)
    ├── graph_mutate(cypher, user_id)             → Neo4j (write)
    ├── sql_query(sql, params)                    → SQLite (read-only)
    ├── sql_mutate(sql, params)                   → SQLite (write)
    └── embed(text)                               → Ollama embedding
```

### Example Interactions

**"Steven has a girlfriend called Mirjam"**
Agent: vector_search("Steven girlfriend") → no match → vector_store("Steven has a girlfriend called Mirjam") → async graph creates `steven --girlfriend--> mirjam` → "Stored: Steven has a girlfriend called Mirjam"

**"Delete all info about dark mode"**
Agent: vector_search("dark mode") → finds 3 facts → vector_delete([id1, id2, id3]) → graph_query(MATCH dark_mode...) → graph_mutate(DELETE) → "Deleted 3 memories and 2 graph relationships about dark mode"

**"What GPU does Steven use?"**
Agent: vector_search("steven GPU") → graph_query(steven → gpu) → "Steven uses an RTX 4090 with 24GB VRAM"

### MCP Interface

```
# v2 (single tool)
memory_agent(request: str) → str

# v1 (still available)
search_memory, add_memories, conversation_memory,
delete_memories, delete_all_memories, get_related_memories, list_memories
```

## Answered Questions (from testing)

- **Model quality**: Yes — qwen3:4b, qwen3.5:4b, and qwen3.5:9b all handle multi-step reasoning. 4b models are adequate; 9b is better for complex multi-search queries.
- **Latency**: Agent loop adds 3-15s depending on complexity. Acceptable for interactive use. v1 tools remain available for latency-sensitive automation.
- **When to use v2 vs v1**: v2 for natural language interaction (chat clients, complex queries). v1 for programmatic/fast operations (auto-recall hooks, bulk operations).
- **Extraction bypass**: Brain's vector_store correctly uses the full extraction pipeline (smart_add_memory with dedup + async graph).

## Not in Scope

- JSON export/import (bulk backup/restore)
- Multi-user (brain per user or shared brain with isolation)
- Memory decay (automatic forgetting of unused/old memories)
- Embeddings migration (switching models without re-indexing)
