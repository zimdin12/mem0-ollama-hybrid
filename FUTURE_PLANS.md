# Future Plans: Memory Brain Agent

## Vision

Replace the current request/response MCP tools with an autonomous **Memory Brain** — an
LLM agent that sits between the caller and the three databases (Qdrant, Neo4j, SQLite).
The caller talks to the brain in natural language. The brain decides what tool calls to
make, how many, and synthesizes the results.

## Current Architecture (v1)

```
Main LLM (Claude, Open WebUI, etc.)
    │
    ├─ add_memories("fact1\nfact2\nfact3")     → regex split → dedup → store
    ├─ conversation_memory(user_msg, llm_resp)  → regex → LLM review → dedup → store
    ├─ search_memory("query", offset=0)         → embedding → Qdrant + Neo4j + SQLite
    ├─ delete_memories(ids)                     → SQLite + Qdrant delete
    └─ get_related_memories("entity")           → Neo4j traversal
```

The main LLM decides WHEN and HOW to call each tool. It formats facts, chooses queries,
manages pagination. The memory system is a passive storage layer.

## Proposed Architecture (v2): Memory Brain

```
Main LLM (Claude, Open WebUI, etc.)
    │
    └─ memory_brain("delete everything about steven")
           │
           Memory Brain LLM (agent loop)
               │
               ├─ Tool: vector_search(query, filters)  → Qdrant
               ├─ Tool: vector_delete(ids)              → Qdrant
               ├─ Tool: vector_store(text, embedding)   → Qdrant
               ├─ Tool: graph_query(cypher)             → Neo4j
               ├─ Tool: graph_mutate(cypher)            → Neo4j
               ├─ Tool: sql_query(sql)                  → SQLite
               ├─ Tool: sql_mutate(sql)                 → SQLite
               └─ Tool: embed(text)                     → Ollama embedding
               │
               (Brain makes as many tool calls as needed,
                then returns synthesized answer)
```

### What Changes

| Aspect | v1 (Current) | v2 (Memory Brain) |
|--------|-------------|-------------------|
| **Who decides what to do** | Main LLM | Memory Brain LLM |
| **Natural language input** | No (structured tool params) | Yes ("what hobbies does steven have?") |
| **Multi-step operations** | Main LLM chains calls | Brain does it internally |
| **Cross-DB operations** | Separate tools per DB | Brain queries all DBs as needed |
| **Main LLM token cost** | Higher (formats facts, manages pagination) | Lower (one natural language call) |
| **Latency** | Fast (direct DB access) | Slower (agent loop with LLM reasoning) |

### Example Interactions

**"Delete everything about Steven"**
Brain would:
1. `graph_query("MATCH (n) WHERE n.name CONTAINS 'steven' RETURN n")` → find all Steven nodes
2. `graph_mutate("MATCH (n)-[r]-() WHERE n.name CONTAINS 'steven' DELETE r, n")` → delete graph
3. `vector_search("steven", limit=100)` → find all Steven vectors
4. `vector_delete([id1, id2, ...])` → delete vectors
5. `sql_query("SELECT id FROM memories WHERE content LIKE '%steven%'")` → find SQLite entries
6. `sql_mutate("UPDATE memories SET state='deleted' WHERE ...")` → mark deleted
7. Return: "Deleted 47 memories, 12 graph nodes, and 23 relationships related to Steven."

**"What hobbies does Steven have?"**
Brain would:
1. `vector_search("steven hobbies interests", limit=10)` → semantic matches
2. `graph_query("MATCH (s)-[r]->(h) WHERE s.name CONTAINS 'steven' AND type(r) IN ['has_hobby', 'enjoys', 'interested_in'] RETURN h.name, type(r)")` → graph relationships
3. `vector_search("steven gaming programming", limit=5)` → follow-up based on graph hints
4. Synthesize: "Steven enjoys game development (working on Echoes of the Fallen in UE5),
   systems programming with Rust, and experimenting with Zig. He prefers local-first
   open-source tools."

**"Remember that I switched from UE5 to Godot"**
Brain would:
1. `vector_search("steven game engine unreal", limit=5)` → find old facts
2. `vector_delete([ids of UE5-related facts])` → remove outdated
3. `graph_mutate("MATCH (s)-[r]->(u) WHERE s.name='steven' AND u.name='unreal_engine_5' DELETE r")` → update graph
4. `embed("Steven switched from Unreal Engine 5 to Godot for game development")` → create embedding
5. `vector_store(text, embedding)` → store new fact
6. `graph_mutate("CREATE (s)-[:uses]->(g) WHERE s.name='steven' AND g.name='godot'")` → update graph
7. Return: "Updated: removed 3 UE5 references, added Godot as Steven's current engine."

## Implementation Considerations

### Model Requirements
- The brain LLM needs reliable **tool calling** (not just text generation)
- qwen3.5:4b handles extraction well but may struggle with multi-step agent reasoning
- Plan: use **qwen3.5:9b** on Ollama for the brain agent — better reasoning at moderate
  VRAM cost (~6GB), benchmarked as solid all-rounder with perfect dedup
- Alternatively: structured agent loop (not native tool_calls) where the code parses
  JSON actions from the LLM and executes them

### Agent Loop Design
```python
class MemoryBrain:
    def __init__(self, llm, tools):
        self.llm = llm          # Ollama LLM for reasoning
        self.tools = tools      # Dict of callable tools
        self.max_steps = 10     # Safety limit on tool calls

    def process(self, user_request: str, context: list = None) -> str:
        messages = [
            {"role": "system", "content": BRAIN_SYSTEM_PROMPT},
            *context or [],
            {"role": "user", "content": user_request}
        ]

        for step in range(self.max_steps):
            response = self.llm.chat(messages, tools=self.tools)

            if response.is_final_answer:
                return response.content

            # Execute tool calls
            for tool_call in response.tool_calls:
                result = self.tools[tool_call.name].execute(tool_call.args)
                messages.append({"role": "tool", "content": result})

        return "Reached maximum steps without completing."
```

### Tools the Brain Would Have

| Tool | Input | Output | Database |
|------|-------|--------|----------|
| `vector_search` | query string, limit, filters | scored results | Qdrant |
| `vector_store` | text (auto-embeds) | stored ID | Qdrant |
| `vector_delete` | list of IDs | count deleted | Qdrant |
| `graph_query` | Cypher query (read-only) | nodes/relationships | Neo4j |
| `graph_mutate` | Cypher query (write) | affected count | Neo4j |
| `sql_query` | SQL query (read-only) | rows | SQLite |
| `sql_mutate` | SQL query (write) | affected count | SQLite |
| `embed` | text | embedding vector | Ollama |

### Safety Guards
- `graph_mutate` and `sql_mutate` require explicit user confirmation for destructive ops
- `max_steps` limit prevents infinite loops
- Read-only tools (search, query) are always safe
- All mutations logged with before/after state

### MCP Interface

The brain would be exposed as a single MCP tool (or a small set):

```
# Option A: Single tool (simplest)
memory_brain(request: str, context?: str) → str

# Option B: Two tools (separates read/write for permissions)
memory_ask(question: str) → str        # read-only, always safe
memory_do(instruction: str) → str      # read-write, may ask confirmation
```

### Migration Path (v1 → v2)

1. **Keep v1 tools working** — don't break existing integrations
2. **Add brain as optional new tool** alongside existing tools
3. **Brain internally uses the same DB access** as v1 tools
4. **Gradually migrate** callers from v1 tools to brain
5. **Eventually** v1 tools become internal (brain-only), brain is the public API

### Open Questions

- **Model size**: Can a 4B model handle multi-step tool calling reliably? Or do we need 7-8B?
- **Latency**: Agent loop with multiple LLM calls could take 2-5 seconds. Acceptable?
- **Cost**: Running a larger model just for memory. Worth it vs current regex approach?
- **Scope**: Should the brain also handle graph extraction (currently async background)?
  Or keep that separate?
- **Context**: Should the brain maintain conversation context across calls within a session?
  Or stay stateless like v1?
- **Confirmation UX**: How to handle destructive operations? Return a confirmation prompt
  and wait for the caller to re-invoke with `confirmed: true`?

## Not in Scope (Separate Features)

- **JSON export/import** — bulk backup and restore of all memories
- **Multi-user** — brain per user or shared brain with user isolation
- **Memory decay** — automatic forgetting of unused/old memories
- **Embeddings migration** — switching embedding models without re-indexing everything
