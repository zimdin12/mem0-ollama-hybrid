# OpenMemory — Local Hybrid Memory System

A mem0 fork that runs 100% locally using any OpenAI-compatible LLM backend (llama.cpp, Ollama, etc.). Provides hybrid memory (vector + graph + temporal) for AI assistants.

**Branch: `memory_agent` (v2)** — Single `memory_agent` MCP tool powered by an autonomous brain agent.

## How to Install for Claude Code

### 1. MCP Server (gives Claude the memory agent tool)

**Option A — SSE (recommended, no dependencies):**

```bash
claude mcp add openmemory --transport sse http://localhost:8765/mcp/claude-code/sse/steven
```

**Option B — Stdio (requires Node.js):**

```bash
cd mcp-server && npm install
claude mcp add openmemory -- node mcp-server/server.js
```

### 2. Skill (tells Claude when to use the memory agent)

Copy the skill file into Claude Code's skill discovery directory:

```bash
# Project-level (this repo only)
mkdir -p .claude/skills/openmemory
cp mcp-server/SKILL.md .claude/skills/openmemory/SKILL.md

# Or personal (all projects)
mkdir -p ~/.claude/skills/openmemory
cp mcp-server/SKILL.md ~/.claude/skills/openmemory/SKILL.md
```

Claude Code auto-discovers skills from `.claude/skills/` — no CLI command needed.

### Prerequisites

The OpenMemory API must be running on port 8765. If it's not running yet, see the Deployment section in README.md.

## MCP Tool

| Tool | Description |
|------|-------------|
| `memory_agent` | Natural language memory operations — search, store, delete, update, explore relationships. The brain agent autonomously chains database tools across vector, graph, and SQL stores. |

**Examples:**
- `"What GPU does Steven use?"` — searches vector + graph, synthesizes answer
- `"Steven has a girlfriend called Mirjam"` — dedup check, store, graph extraction
- `"Steven switched from UE5 to Godot"` — finds old fact, deletes it, stores correction
- `"Delete all memories about dark mode"` — searches, batch deletes
- `"How is Steven connected to Echoes of the Fallen?"` — graph traversal

## Project Structure

- `openmemory/` — Docker services: FastAPI backend (port 8765) + Next.js UI (port 3000/3100)
- `openmemory/api/app/brain/` — Brain agent (prompts, tools, agent loop)
- `mcp-server/` — Host-side MCP server (Node.js stdio) + Claude Code skill
- `mem0/` — Core mem0 Python library (modified from upstream)
- See README.md for full deployment and environment variable docs
- See FORK_CHANGES.md for detailed technical changelog vs upstream
