# OpenMemory — Hybrid Memory System

You have access to a persistent hybrid memory system (vector search + knowledge graph) via MCP tools. Use it to remember and recall information across sessions.

## Available Tools

| Tool | When to use |
|------|------------|
| `mem_search` | Recall facts, preferences, past decisions, people, projects, or any previously stored context |
| `mem_store` | Save important facts, preferences, project decisions, or anything worth remembering |
| `mem_forget` | Delete a memory that is outdated, incorrect, or no longer relevant |
| `mem_related` | Explore connections between people, projects, and concepts via graph traversal |

## Auto-Recall (start of conversation)

At the **start of every conversation**, search memory for context relevant to the current task:

1. If the user mentions a project, person, or topic — `mem_search` for it
2. If working in a specific codebase — `mem_search` for the project name or directory
3. If the user references a past decision or preference — `mem_search` before assuming

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

1. `mem_search` to find the outdated memory
2. `mem_forget` the old one (by ID from search results)
3. `mem_store` the corrected version

mem0 has built-in deduplication, but explicit forget + store is more reliable for corrections.

## Graph Traversal

Use `mem_related` when exploring connections:

- "What do we know about Steven?" → `mem_related` with entity "Steven"
- "What tools are used in this project?" → `mem_related` with project name
- Following up on a search result that mentions an entity → `mem_related` to see its connections
