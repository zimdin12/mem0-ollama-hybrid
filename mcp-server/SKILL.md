---
name: openmemory
description: Hybrid memory system (vector + graph + temporal). Use search_memory, add_memories, delete_memories, get_related_memories tools to recall, store, and explore persistent knowledge.
---

# OpenMemory — Persistent Memory for Claude Code

Hybrid memory system across sessions. Stores facts, preferences, project context, people, and decisions in vector + graph databases. Use the direct tools below — they are fast (~100ms search, ~1s store).

## Tools (use these)

| Tool | When |
|------|------|
| `mem_search` | Recall facts. Start of conversation, when user references past context, or when you need project/person info. |
| `mem_store` | Save durable facts: preferences, project decisions, environment details, people, lessons learned. |
| `mem_forget` | Delete outdated or incorrect memories by ID. |
| `mem_related` | Explore entity connections in the knowledge graph. |

## When to Search

- **Start of conversation**: search for context about the current project or task
- **User says "remember"/"recall"/"we discussed"**: search for what they're referring to
- **Before making assumptions**: check if there's stored context about the user's setup

## When to Store

Store facts that would be useful in **future sessions**:
- User preferences and decisions ("Steven prefers Vite over CRA")
- Project architecture ("The app uses PostgreSQL with PostGIS")
- People and relationships ("Mirjam is Steven's girlfriend")
- Environment facts ("RTX 4090 with 24GB VRAM")
- Lessons learned ("Queue worker needs --timeout=0 for Redis keepalive")

**Don't store**: ephemeral task state, code snippets, things derivable from the codebase.

## How to Store Well

Use **concise, self-contained statements** with specific names:
- Good: `mem_store("Steven prefers TypeScript over JavaScript for new projects")`
- Bad: `mem_store("he likes TS")` — who? compared to what?

## Advanced: Memory Agent

| Tool | When |
|------|------|
| `memory_agent` | Complex operations: corrections ("Steven switched from X to Y"), batch deletes, or when you need the system to reason about what to search/store/update. Slower (~5s) — uses LLM reasoning. |
