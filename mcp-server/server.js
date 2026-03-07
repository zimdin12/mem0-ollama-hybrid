#!/usr/bin/env node

/**
 * OpenMemory MCP Server
 *
 * Exposes hybrid vector + graph memory (mem0) as MCP tools.
 * Designed for Claude Code, OpenCode, or any MCP-compatible client.
 *
 * Tools: mem_search, mem_store, mem_forget, mem_related
 *
 * Config via env vars:
 *   MEMORY_API_URL  — OpenMemory API base URL (default: http://localhost:8765)
 *   MEMORY_USER_ID  — User ID for memory operations (default: steven)
 *   MEMORY_APP_NAME — App name tag for stored memories (default: claude-code)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_URL = (process.env.MEMORY_API_URL || "http://localhost:8765").replace(/\/$/, "");
const USER_ID = process.env.MEMORY_USER_ID || "steven";
const APP_NAME = process.env.MEMORY_APP_NAME || "claude-code";

// ---------------------------------------------------------------------------
// API helper
// ---------------------------------------------------------------------------

async function api(path, { method = "GET", body, params } = {}) {
  let url = `${API_URL}${path}`;
  if (params) {
    const qs = new URLSearchParams(params).toString();
    if (qs) url += `?${qs}`;
  }

  const resp = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`OpenMemory ${method} ${path}: ${resp.status} ${text}`);
  }

  const ct = resp.headers.get("content-type") || "";
  return ct.includes("application/json") ? resp.json() : resp.text();
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

const TOOLS = [
  {
    name: "mem_search",
    description:
      "Search the hybrid memory system (vector + graph). Use this to recall facts, preferences, past decisions, people, projects, or any previously stored context.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Semantic search query" },
        limit: { type: "number", description: "Max results (default 5)" },
      },
      required: ["query"],
    },
  },
  {
    name: "mem_store",
    description:
      "Store a memory. mem0 auto-extracts entities and relationships for the knowledge graph. Use for facts, preferences, project decisions, or anything worth remembering across sessions.",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string", description: "Memory text — concise and self-contained" },
        metadata: { type: "object", description: 'Optional metadata, e.g. {"topic": "preferences"}' },
      },
      required: ["text"],
    },
  },
  {
    name: "mem_forget",
    description: "Delete a memory by ID. Use when a memory is outdated or incorrect.",
    inputSchema: {
      type: "object",
      properties: {
        memory_id: { type: "string", description: "Memory ID to delete" },
      },
      required: ["memory_id"],
    },
  },
  {
    name: "mem_related",
    description:
      "Get related memories and entities via graph traversal. Use to explore connections between people, projects, and concepts.",
    inputSchema: {
      type: "object",
      properties: {
        entity: { type: "string", description: "Entity name (person, project, concept)" },
      },
      required: ["entity"],
    },
  },
  {
    name: "memory_ask",
    description:
      "Ask a natural language question about stored memories. The Memory Brain agent autonomously searches vector store, knowledge graph, and SQLite to synthesize a comprehensive answer. Use for complex queries like: 'what hobbies does Steven have?', 'summarize what you know about project X'.",
    inputSchema: {
      type: "object",
      properties: {
        request: { type: "string", description: "Natural language question about memories" },
      },
      required: ["request"],
    },
  },
  {
    name: "memory_do",
    description:
      "Perform a natural language memory operation: store, update, delete, or reorganize memories. The Memory Brain agent autonomously decides which tools to use. Examples: 'remember that Steven switched to Godot', 'delete all memories about dark mode'.",
    inputSchema: {
      type: "object",
      properties: {
        request: { type: "string", description: "Natural language instruction for memory operation" },
      },
      required: ["request"],
    },
  },
];

// ---------------------------------------------------------------------------
// Tool handlers
// ---------------------------------------------------------------------------

async function handleSearch({ query, limit = 5 }) {
  // Vector search
  const result = await api("/api/v1/memories/", {
    params: { user_id: USER_ID, search_query: query, size: String(limit) },
  });
  const memories = result?.items || [];

  // Graph context (optional, don't fail if unavailable)
  let graphText = "";
  try {
    const graph = await api(`/api/v1/memories/graph/context/${encodeURIComponent(query)}`, {
      params: { user_id: USER_ID },
    });
    const parts = [];
    if (graph?.root_entities?.length) {
      parts.push(`Entities: ${graph.root_entities.join(", ")}`);
    }
    if (graph?.related_entities) {
      for (const [dist, entities] of Object.entries(graph.related_entities)) {
        parts.push(`Distance ${dist}: ${entities.map((e) => `${e.name} (${e.type})`).join(", ")}`);
      }
    }
    if (graph?.memories?.length) {
      parts.push("Graph memories:");
      for (const m of graph.memories.slice(0, 3)) {
        parts.push(`- ${m.content.slice(0, 200)}`);
      }
    }
    if (parts.length) graphText = parts.join("\n");
  } catch {
    // graph search is optional
  }

  if (!memories.length && !graphText) {
    return "No memories found. Use mem_store to save new memories.";
  }

  let text = "";
  if (memories.length) {
    text += memories
      .map((m, i) => {
        const cats = m.categories?.length ? ` [${m.categories.join(", ")}]` : "";
        const id = m.id ? ` (id: ${m.id})` : "";
        return `### Memory ${i + 1}${cats}${id}\n${m.content}`;
      })
      .join("\n\n");
  }
  if (graphText) {
    text += `\n\n### Graph Context\n${graphText}`;
  }
  return text;
}

async function handleStore({ text, metadata = {} }) {
  const result = await api("/api/v1/memories/", {
    method: "POST",
    body: { text, user_id: USER_ID, metadata, app: APP_NAME },
  });

  const added = result?.results?.filter((r) => r.event?.toLowerCase() === "add") || [];
  const updated = result?.results?.filter((r) => r.event?.toLowerCase() === "update") || [];

  let msg = "Memory processed.";
  if (added.length) msg += ` Added: ${added.length}`;
  if (updated.length) msg += ` Updated: ${updated.length}`;
  return msg;
}

async function handleForget({ memory_id }) {
  await api("/api/v1/memories/", {
    method: "DELETE",
    body: { memory_ids: [memory_id], user_id: USER_ID },
  });
  return `Memory ${memory_id} deleted.`;
}

async function handleRelated({ entity }) {
  const result = await api(
    `/api/v1/memories/graph/entity/${encodeURIComponent(entity)}/related`,
    { params: { user_id: USER_ID } },
  );

  const rels = result?.relationships || [];
  if (!rels.length) return `No relationships found for "${entity}".`;

  let text = `### Relationships for "${entity}" (${rels.length})\n\n`;
  text += rels
    .map((r) => `- **${r.source}** —${r.relationship || "related"}→ **${r.target}**${r.distance > 1 ? ` (${r.distance} hops)` : ""}`)
    .join("\n");
  return text;
}

async function handleMemoryAsk({ request }) {
  const result = await api("/api/v1/brain/ask", {
    method: "POST",
    body: { request, user_id: USER_ID },
  });
  return result?.answer || JSON.stringify(result);
}

async function handleMemoryDo({ request }) {
  const result = await api("/api/v1/brain/do", {
    method: "POST",
    body: { request, user_id: USER_ID, confirmed: true },
  });
  return result?.answer || JSON.stringify(result);
}

const HANDLERS = {
  mem_search: handleSearch,
  mem_store: handleStore,
  mem_forget: handleForget,
  mem_related: handleRelated,
  memory_ask: handleMemoryAsk,
  memory_do: handleMemoryDo,
};

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

const server = new Server(
  { name: "openmemory", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const handler = HANDLERS[name];
  if (!handler) {
    return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
  }

  try {
    const result = await handler(args || {});
    return { content: [{ type: "text", text: result }] };
  } catch (err) {
    return { content: [{ type: "text", text: `Error: ${err.message}` }], isError: true };
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const transport = new StdioServerTransport();
await server.connect(transport);
