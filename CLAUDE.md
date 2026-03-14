# OpenMemory — Local Hybrid Memory System

A mem0 fork that runs 100% locally using any OpenAI-compatible LLM backend (llama.cpp, Ollama, etc.). Provides hybrid memory (vector + graph + temporal) for AI assistants.

**Branch: `memory_agent` (v2)** — Full MCP toolset + autonomous brain agent.

## How to Connect with Claude Code

### Prerequisites

The OpenMemory API must be running on port 8765 with its dependencies (Qdrant, Neo4j, an LLM backend for extraction, an embedding model). See the Deployment section in README.md.

### Step 1 — Register the MCP Server

Add the SSE MCP server globally so it's available in all projects:

**Option A — Global (recommended):**

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "openmemory": {
      "type": "sse",
      "url": "http://localhost:8765/mcp/claude-code/sse/YOUR_USER_ID"
    }
  }
}
```

Replace `YOUR_USER_ID` with your user ID (e.g. `steven`). The user ID scopes all memory operations.

**Option A (CLI shorthand):**

```bash
claude mcp add openmemory --transport sse http://localhost:8765/mcp/claude-code/sse/YOUR_USER_ID -s user
```

The `-s user` flag makes it global (all projects). Without it, it's project-scoped only.

**Option B — Stdio (alternative, requires Node.js):**

```bash
cd mcp-server && npm install
claude mcp add openmemory -s user -- node mcp-server/server.js
```

### Step 2 — Install the Skill

The skill file tells Claude Code **when** and **how** to use the memory tools:

```bash
mkdir -p ~/.claude/skills/openmemory
cp mcp-server/SKILL.md ~/.claude/skills/openmemory/SKILL.md
```

Claude Code auto-discovers skills from `~/.claude/skills/`.

### Step 3 — Add Instructions to CLAUDE.md

Add the following to your global `~/.claude/CLAUDE.md` so Claude knows to use memory proactively:

```markdown
## OpenMemory — Persistent Memory System

You have access to a hybrid memory system (vector + graph + temporal) via the `openmemory` MCP server. This stores facts, preferences, project context, and decisions across all sessions and projects.

### How to use it

- **Start of conversation**: Use `search_memory` to recall context about the user, current project, or task.
- **During conversation**: When the user shares durable facts, use `add_memories` to store them.
- **Before compaction**: Save important conversation state to memory before your context window refreshes.
- **Corrections**: Use `memory_agent` for complex operations like "Steven switched from X to Y".

### When the memory system is unavailable

If a memory tool call fails or times out:
- Do NOT keep retrying — tell the user the memory system appears to be offline.
- Continue working normally without memory.
- When the user says they reconnected/started the memory system, resume using memory tools.

### What to store vs skip

**Store**: user preferences, project architecture, people/relationships, environment facts, lessons learned, decisions.
**Skip**: ephemeral task state, code being edited, things derivable from git/codebase, temporary debugging info.
```

### Verify It Works

Start the OpenMemory stack, then in any project:

```
You: search my memory for what projects I'm working on
Claude: [calls search_memory] Found 5 memories...
```

If you see "MCP server openmemory is not connected", the API isn't running. Start the stack and tell Claude to try again.

## MCP Tools (via SSE)

The SSE endpoint exposes these tools:

| Tool | Description |
|------|-------------|
| `search_memory` | Hybrid search (vector + graph + temporal). Returns ranked results with scores and sources. |
| `add_memories` | Store facts. One per line, self-contained. Auto-extracts entities for knowledge graph. Auto-deduplicates. |
| `conversation_memory` | Extract facts from a chat turn (user message + assistant response). |
| `delete_memories` | Delete by ID. Shows related memories that may also need review. |
| `get_related_memories` | Graph traversal — explore connections between entities. |
| `memory_agent` | Natural language brain agent. Chains search/store/delete autonomously. Slower (~5s). |
| `memory_status` | System health check: store status, memory counts, last update. |

**Examples:**
- `search_memory("Vulkan shadow performance")` → returns shadow map facts with scores
- `add_memories("Steven uses PostgreSQL with PostGIS for geospatial queries.")` → extracts, deduplicates, stores
- `conversation_memory(user_msg, assistant_msg)` → auto-extracts facts from the exchange
- `memory_agent("Steven switched from UE5 to Godot")` → finds old fact, deletes it, stores correction

## Project Structure

- `openmemory/` — Docker services: FastAPI backend (port 8765) + Next.js UI (port 3000/3100)
- `openmemory/api/app/mcp_server.py` — SSE MCP server (tools + endpoints)
- `openmemory/api/app/utils/enhanced_memory.py` — Hybrid search, smart extraction, scoring
- `openmemory/api/app/brain/` — Brain agent (prompts, tools, agent loop)
- `mcp-server/` — Host-side MCP server (Node.js stdio alternative) + skill file
- `mem0/` — Core mem0 Python library (modified from upstream)
- See README.md for full deployment and environment variable docs
- See FORK_CHANGES.md for detailed technical changelog vs upstream
