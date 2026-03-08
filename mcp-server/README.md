# OpenMemory MCP Server + Skill

MCP server that exposes the OpenMemory hybrid memory system (vector + graph via mem0) to any MCP client. Includes a Claude Code skill for auto-recall/capture behavior.

## Tools

| Tool | Description |
|------|-------------|
| `search_memory` | Hybrid search across vector + graph + temporal (paginated) |
| `add_memories` | Smart add with dedup — only stores truly new information |
| `list_memories` | List all memories with permission filtering |
| `delete_memories` | Delete specific memories by ID |
| `delete_all_memories` | Delete all memories for the user |
| `conversation_memory` | Extract and store facts from a conversation turn |
| `get_related_memories` | Explore entity relationships via graph traversal |

## Setup

```bash
cd mcp-server
npm install
```

## Add to Claude Code

Two steps — MCP server (tools) and skill (behavior) are registered separately.

### 1. Register MCP server (provides the tools)

```bash
claude mcp add openmemory -- node /path/to/mem0-fork/mcp-server/server.js
```

Or manually add to `~/.claude.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "openmemory": {
      "type": "stdio",
      "command": "node",
      "args": ["/path/to/mem0-fork/mcp-server/server.js"],
      "env": {
        "MEMORY_API_URL": "http://localhost:8765",
        "MEMORY_USER_ID": "steven"
      }
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

Add to your OpenCode MCP config:

```json
{
  "openmemory": {
    "type": "stdio",
    "command": "node",
    "args": ["/path/to/mem0-fork/mcp-server/server.js"],
    "env": {
      "MEMORY_API_URL": "http://localhost:8765",
      "MEMORY_USER_ID": "steven"
    }
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
