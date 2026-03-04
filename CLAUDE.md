# OpenMemory — Local Hybrid Memory System (Ollama Fork)

This is a mem0 fork that runs 100% locally using Ollama. It provides hybrid memory (vector + graph) for AI assistants.

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

## Tools Available After Setup

| Tool | Description |
|------|-------------|
| `mem_search` | Search memories (vector + graph hybrid) |
| `mem_store` | Store a memory (auto-extracts entities) |
| `mem_forget` | Delete a memory by ID |
| `mem_related` | Explore entity relationships via graph |

## Project Structure

- `openmemory/` — Docker services: FastAPI backend (port 8765) + Next.js UI (port 3000/3100)
- `mcp-server/` — Host-side MCP server (Node.js stdio) + Claude Code skill
- `mem0/` — Core mem0 Python library (from upstream)
- See README.md for full deployment and environment variable docs
