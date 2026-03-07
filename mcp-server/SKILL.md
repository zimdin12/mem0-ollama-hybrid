---
name: openmemory
description: Memory Agent — talk to the memory system in natural language. It autonomously searches, stores, deletes, and updates memories across vector, graph, and temporal databases.
---

# OpenMemory — Memory Agent (v2)

You have access to a Memory Agent — an autonomous LLM that manages a persistent hybrid memory system (vector + graph + temporal). Talk to it in natural language. It determines intent and chains database operations as needed.

## MCP Tool

| Tool | When to use |
|------|------------|
| `memory_agent` | All memory operations. Search, store, delete, update, explore relationships — describe what you need in natural language. |

**Examples:**
- Search: "What GPU does Steven use?", "What do we know about the Echoes project?"
- Store: "Steven has a girlfriend called Mirjam", "The project uses PostgreSQL for production"
- Update: "Steven switched from UE5 to Godot", "Steven upgraded to RTX 5090"
- Delete: "Delete all memories about dark mode", "Remove the memory about PHP"
- Explore: "How is Steven connected to Echoes of the Fallen?", "What tools does Steven use?"
- Complex: "Tell me everything about Steven's hardware setup"

## When to Use Memory

### Start of Conversation
Ask the memory agent about context relevant to the current task:
- "What do we know about Steven's preferences?"
- "What's stored about this project?"

### During Conversation
When the user shares durable information — facts useful in future sessions:
- Preferences, project decisions, environment facts, people, lessons learned

Tell the memory agent in natural language: "Remember that Steven prefers dark mode in all editors"

### Corrections and Updates
Tell the agent directly: "Steven no longer uses Godot, he switched to Unreal Engine 5"

The agent will search for the old fact, delete it, and store the updated version.

## How to Store Well

When telling the agent to remember something, use **concise, self-contained statements** with specific names:

- Good: "Remember that Steven prefers TypeScript over JavaScript for new projects"
- Bad: "Remember he likes TS" (who? compared to what?)
- Good: "Remember that Echoes of the Fallen uses dual contouring for terrain"
- Bad: "Remember it uses dual contouring" (what project?)

## Conversation Memory Mode

When the user says **"use conversation memory"**, call `memory_agent` after every turn:
- "Process this conversation: user said '...', assistant responded '...'"
- The agent extracts facts, deduplicates, and stores automatically
- Continue until the user says to stop
