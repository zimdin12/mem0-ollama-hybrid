---
name: openmemory
description: Hybrid memory system (vector + graph + temporal). Use search_memory, add_memories, delete_memories, get_related_memories tools to recall, store, and explore persistent knowledge.
---

# OpenMemory — Hybrid Memory System

You have access to a persistent hybrid memory system via MCP tools. It combines vector search (semantic similarity), graph traversal (entity relationships), and temporal context (recency) to provide rich recall.

## Available MCP Tools

| Tool | When to use |
|------|------------|
| `search_memory` | Recall facts, preferences, decisions, people, projects — any previously stored context. Use `offset` to paginate. |
| `add_memories` | Save pre-formatted facts (one per line, self-contained with subject). Best for bulk storage. |
| `conversation_memory` | Process a conversation turn — regex extracts candidates, LLM reviews for quality. Best for natural conversation. |
| `delete_memories` | Delete outdated or incorrect memories by ID |
| `delete_all_memories` | Wipe all memories for the user (use with extreme caution) |
| `get_related_memories` | Explore connections between entities via graph traversal |
| `list_memories` | List all stored memories with optional filtering |

## Usage Modes

### Default Mode
Use memory tools when relevant — search before answering questions about the user or past decisions, save when new durable facts appear. No need to call memory on every turn.

### Conversation Memory Mode
When the user says **"use conversation memory"** (or similar), switch to continuous mode:
- Call `conversation_memory` after **every turn**, passing your user_message, llm_response, and optionally recent_context (last 2-3 turns as JSON array of `{role, content}` objects)
- The system handles extraction, LLM review, dedup, and storage automatically
- Continue until the user says to stop

### When to use which storage tool
- **`add_memories`**: You have clean, pre-formatted facts (one per line). Best for bulk storage where you've already done the thinking.
- **`conversation_memory`**: Raw conversation — let the system extract and LLM-review facts. Best for natural conversation where you don't want to manually format.

## Auto-Recall (start of conversation)

At the **start of every conversation**, search memory for context relevant to the current task:

1. If the user mentions a project, person, or topic — `search_memory` for it
2. If working in a specific codebase — `search_memory` for the project name or directory
3. If the user references a past decision or preference — `search_memory` before assuming

Keep searches focused. One or two targeted queries are better than broad sweeps.

## Auto-Capture (during conversation)

Store memories when the user shares **durable information** — facts that will be useful in future sessions:

- **Preferences**: "I prefer tabs over spaces", "always use pnpm", "I like minimal comments"
- **Project decisions**: "we chose Postgres over SQLite for production", "auth uses JWT"
- **Environment facts**: "my GPU is an RTX 4090", "I run Docker on Windows with WSL2"
- **People and roles**: "Alice is the backend lead", "Bob handles deployments"
- **Lessons learned**: "the Neo4j APOC plugin needs explicit enabling", "qwen3 thinking mode breaks tool calls"

**Do NOT store**:
- Ephemeral task context (current file being edited, in-progress debugging)
- Information already in project docs or README files
- Speculative or unverified conclusions
- Anything the user asks you not to remember

## How to Store Well

Write memories as **concise, self-contained facts**. Each memory should make sense on its own without context:

- Good: "Steven prefers TypeScript over JavaScript for new projects"
- Bad: "He likes TS" (who? compared to what?)
- Good: "OpenClaw project uses qwen3-coder-next as the default LLM via LM Studio"
- Bad: "The model was changed" (which model? changed from what?)

## Updating and Correcting

When the user corrects a previous fact or preference:

1. `search_memory` to find the outdated memory
2. `delete_memories` the old one (by ID from search results)
3. `add_memories` the corrected version

The system has built-in deduplication, but explicit delete + add is more reliable for corrections.

## Graph Traversal

Use `get_related_memories` when exploring connections:

- "What do we know about Steven?" → `get_related_memories` with entity "Steven"
- "What tools are used in this project?" → `get_related_memories` with project name
- Following up on a search result that mentions an entity → explore its connections

## Search Results

Search results include:
- **Score**: relevance percentage (higher = more relevant)
- **Source**: where the result came from (vector, graph, or temporal)
- **ID**: use this for deletion or follow-up queries

Results are ranked by a hybrid of semantic similarity, graph relationships, and recency.
