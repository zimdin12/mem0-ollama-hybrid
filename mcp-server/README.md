# OpenMemory MCP Server + Skill

MCP server that exposes the OpenMemory hybrid memory system (vector + graph via mem0) to any MCP client. Includes a Claude Code skill for auto-recall/capture behavior.

**Branch: `memory_agent` (v2)** — Single `memory_agent` tool powered by an autonomous brain agent.

## Tool

| Tool | Description |
|------|-------------|
| `memory_agent` | Natural language memory operations — search, store, delete, update, explore. The brain agent chains database tools autonomously. |

## Setup

```bash
cd mcp-server
npm install
```

## Add to Claude Code

Two steps — MCP server (tool) and skill (behavior) are registered separately.

### 1. Register MCP server (provides the tool)

**SSE (recommended — no Node.js needed):**

```bash
claude mcp add openmemory --transport sse http://localhost:8765/mcp/claude-code/sse/steven
```

**Stdio (alternative):**

```bash
claude mcp add openmemory -- node /path/to/mem0-fork/mcp-server/server.js
```

Or manually add to `~/.claude.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "openmemory": {
      "type": "sse",
      "url": "http://localhost:8765/mcp/claude-code/sse/steven"
    }
  }
}
```

### 2. Install skill (adds auto-recall/capture behavior)

Copy `SKILL.md` into Claude Code's skill discovery directory:

```bash
# Project-level
mkdir -p .claude/skills/openmemory
cp SKILL.md .claude/skills/openmemory/SKILL.md

# Or personal (all projects)
mkdir -p ~/.claude/skills/openmemory
cp SKILL.md ~/.claude/skills/openmemory/SKILL.md
```

## Add to OpenCode

```json
{
  "openmemory": {
    "type": "sse",
    "url": "http://localhost:8765/mcp/opencode/sse/steven"
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_API_URL` | `http://localhost:8765` | OpenMemory API URL |
| `MEMORY_USER_ID` | `steven` | User ID for memory operations |
| `MEMORY_APP_NAME` | `claude-code` | App tag for stored memories |

## Requirements

- Node.js 18+
- OpenMemory API running (the `openmemory-mcp` Docker container)
