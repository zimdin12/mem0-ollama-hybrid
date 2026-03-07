# Memory Brain Agent

An autonomous LLM agent that sits between callers and the three databases (Qdrant, Neo4j, SQLite). Accepts natural-language requests, decides which tools to call, and returns synthesized answers.

## Architecture

```
Caller (Claude Code, OpenClaw, REST API, MCP client)
    │
    └─ "What hobbies does Steven have?"
           │
       Memory Brain Agent (qwen3.5:9b via Ollama JSON mode)
           │
           ├─ Step 1: vector_search("steven hobbies interests")     → Qdrant
           ├─ Step 2: graph_query("MATCH (s)-[r]->(h) WHERE ...")   → Neo4j
           ├─ Step 3: vector_search("steven gaming programming")    → Qdrant (follow-up)
           └─ Step 4: final=true, answer="Steven enjoys..."         → Return to caller
```

## Tools (8 primitives)

| Tool | Type | Database | Description |
|------|------|----------|-------------|
| `vector_search` | read | Qdrant | Semantic search over memory facts |
| `vector_store` | write | Qdrant + SQLite | Store a fact (auto-embeds, dedup, triggers graph extraction) |
| `vector_delete` | write | Qdrant + SQLite | Delete memories by ID |
| `graph_query` | read | Neo4j | Read-only Cypher queries |
| `graph_mutate` | write | Neo4j | Write Cypher (CREATE/MERGE/DELETE/SET) |
| `sql_query` | read | SQLite | Read-only SQL queries |
| `sql_mutate` | write | SQLite | Write SQL (INSERT/UPDATE/DELETE) |
| `embed` | read | Ollama | Generate 1024-dim embedding vector |

Read tools are always available. Write tools are blocked in read-only mode (`/brain/ask`).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/brain/ask` | POST | Read-only brain query. Natural language → synthesized answer. |
| `/api/v1/brain/do` | POST | Read-write brain operation. Confirmation gate for destructive ops. |
| `/api/v1/brain/audit` | GET | View brain audit log. |
| `/api/v1/brain/status` | GET | Brain agent configuration and status. |

## MCP Tools

| Tool | Mode | Description |
|------|------|-------------|
| `memory_ask` | read-only | Natural language question about memories |
| `memory_do` | read-write | Natural language memory operation (auto-confirmed via MCP) |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `BRAIN_LLM_MODEL` | `qwen3.5:9b` | Ollama model for the brain agent loop |
| `BRAIN_OLLAMA_URL` | `http://ollama:11434` | Ollama API endpoint |
| `BRAIN_MAX_STEPS` | `12` | Max tool calls per brain invocation |
| `BRAIN_CONFIRM_DESTRUCTIVE` | `true` | Require confirmation for destructive ops via REST |

## Agent Loop Design

The brain uses **Ollama JSON mode** (not native tool_calls). Every turn, the LLM emits:

```json
{
  "thinking": "I need to search for Steven's hobbies first",
  "action": "vector_search",
  "args": {"query": "steven hobbies", "limit": 10, "user_id": "steven"},
  "final": false
}
```

When done:

```json
{
  "thinking": "I have enough information to answer",
  "action": null,
  "args": null,
  "final": true,
  "answer": "Steven enjoys game development with Godot, systems programming with Rust..."
}
```

### Why JSON mode instead of native tool_calls?

Small models (3-9B) have unreliable multi-step native tool calling. JSON mode forces valid JSON output every turn, and the structured schema (thinking/action/args/final) is simple enough for any model to follow reliably.

### Error Handling

- JSON parse failure → retry up to 2 times with a reminder to emit valid JSON
- Tool execution error → append error as tool result, let brain decide how to proceed
- Max steps reached → return partial results with error note
- All invocations logged to `brain_audit` SQLite table

### Confirmation Flow (REST API only)

For destructive operations via `/brain/do`:

1. **First call** (`confirmed=false`): Brain runs in dry-run mode. Destructive tools are collected but not executed. Returns `requires_confirmation=true` with `planned_actions`.
2. **Second call** (`confirmed=true`, include `planned_actions`): Executes the planned actions directly.

MCP tools (`memory_do`) skip confirmation — the calling LLM has already decided to act.

## Audit Trail

Every tool call and brain invocation is logged to the `brain_audit` SQLite table:

```sql
SELECT ts, tool, input, output, error FROM brain_audit ORDER BY id DESC LIMIT 10;
```

View via REST: `GET /api/v1/brain/audit?user_id=steven&limit=50`
