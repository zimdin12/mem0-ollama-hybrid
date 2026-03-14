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
    name: "memory_agent",
    description:
      "Talk to the Memory Agent in natural language. It autonomously searches, stores, deletes, or updates memories across all 3 databases (vector, graph, metadata). Examples: 'What hobbies does Steven have?', 'Steven has a girlfriend called Mirjam', 'Delete all memories about dark mode'. The agent determines intent and chains tool calls as needed.",
    inputSchema: {
      type: "object",
      properties: {
        request: { type: "string", description: "Natural language request for the memory agent" },
      },
      required: ["request"],
    },
  },
];

// ---------------------------------------------------------------------------
// Tool handlers
// ---------------------------------------------------------------------------

async function handleSearch({ query, limit = 5 }) {
  // Hybrid search (vector + graph + temporal) with relevance scoring
  const result = await api("/api/v1/memories/search/", {
    method: "POST",
    body: { query, user_id: USER_ID, limit },
  });

  const memories = result?.results || [];
  if (!memories.length) {
    return "No memories found. Use mem_store to save new memories.";
  }

  return memories
    .map((m, i) => {
      const score = m.score ? ` (${m.score.toFixed(3)})` : "";
      const src = m.source ? ` [${m.source}]` : "";
      const id = m.id ? ` (id: ${m.id})` : "";
      return `${i + 1}. ${m.memory}${score}${src}${id}`;
    })
    .join("\n");
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

async function handleMemoryAgent({ request }) {
  const result = await api("/api/v1/brain", {
    method: "POST",
    body: { request, user_id: USER_ID },
  });
  return result?.answer || JSON.stringify(result);
}

const HANDLERS = {
  mem_search: handleSearch,
  mem_store: handleStore,
  mem_forget: handleForget,
  mem_related: handleRelated,
  memory_agent: handleMemoryAgent,
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
