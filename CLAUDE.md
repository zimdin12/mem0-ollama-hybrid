# OpenMemory — Local Hybrid Memory System (Ollama Fork)

A mem0 fork that runs 100% locally using Ollama. Provides hybrid memory (vector + graph + temporal) for AI assistants.

## How to Install for Claude Code

### 1. MCP Server (gives Claude memory tools)

**Option A — SSE (recommended, no dependencies):**

```bash
claude mcp add openmemory --transport sse http://localhost:8765/mcp/claude-code/sse/steven
```

**Option B — Stdio (requires Node.js):**

```bash
cd mcp-server && npm install
claude mcp add openmemory -- node mcp-server/server.js
```

### 2. Skill (tells Claude when to use memory tools)

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

## MCP Tools Available After Setup

| Tool | Description |
|------|-------------|
| `search_memory` | Hybrid search across vector + graph + temporal (returns 10 results) |
| `add_memories` | Smart add with dedup — only stores truly new information |
| `list_memories` | List all memories with permission filtering |
| `delete_memories` | Delete specific memories by ID |
| `delete_all_memories` | Delete all memories for the user |
| `handle_conversation` | Process user message + LLM response, extract memorable content |
| `get_related_memories` | Explore entity relationships via graph traversal |

## Project Structure

- `openmemory/` — Docker services: FastAPI backend (port 8765) + Next.js UI (port 3000/3100)
- `mcp-server/` — Host-side MCP server (Node.js stdio) + Claude Code skill
- `mem0/` — Core mem0 Python library (modified from upstream)
- `test_memory_system.py` — 5-phase test suite
- See README.md for full deployment and environment variable docs
- See FORK_CHANGES.md for detailed technical changelog vs upstream
