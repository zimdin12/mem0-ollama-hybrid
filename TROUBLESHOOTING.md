# OpenMemory Troubleshooting

Common issues and solutions for the Ollama fork.

## Memory Not Storing

**Symptom**: `add_memories` returns `{"added_memories": [], "skipped_facts": [...]}`

1. **Duplicate detected** — cosine similarity >= 0.95 with existing memory. Check with `search_memory`.
2. **Noise filter** — input classified as greeting/filler. Must be a factual statement.
3. **Too short** — inputs under 10 chars are rejected.

**Symptom**: REST API returns 404 "User not found"

The REST API (not MCP) requires pre-existing users in SQLite. MCP auto-creates users.
Fix: Use MCP tools, or create user first:
```python
from app.database import SessionLocal
from app.utils.db import get_user_and_app
db = SessionLocal()
get_user_and_app(db, "steven", "your-app")
db.close()
```

## Graph Not Populating

**Symptom**: Search returns vector results but no graph results

1. Graph extraction runs async (3-5s after API response). Wait and retry.
2. Check Neo4j is running: `docker exec neo4j cypher-shell -u neo4j -p openmemory "MATCH (n) RETURN count(n)"`
3. Check extraction model is loaded: `docker exec ollama ollama list`
4. Short inputs (<100 chars) bypass extraction — graph only populates for longer content.

## Search Returns Nothing

1. Check Qdrant has vectors: `curl http://localhost:6333/collections/openmemory`
2. Check correct user_id — memories are filtered by user
3. Try different query angle — embedding similarity is directional

## Container Won't Start

**openmemory-mcp fails with model error:**
```
Ollama model not found
```
Pull the required models:
```bash
docker exec ollama ollama pull qwen3:4b-instruct-2507-q4_K_M
docker exec ollama ollama pull qwen3-embedding:0.6b
```

**Neo4j OOM:**
Neo4j needs ~1GB RAM. Check `docker stats neo4j`. Set `NEO4J_server_memory_heap_max__size=512m` if constrained.

## MCP Connection Issues

**SSE transport not connecting:**
1. Verify API is running: `curl http://localhost:8765/api/v1/health`
2. Check SSE URL format: `http://localhost:8765/mcp/{client_name}/sse/{user_id}`
3. Both `client_name` and `user_id` are created automatically on first SSE connection

**Stdio transport fails:**
1. Check Node.js 18+: `node --version`
2. Install deps: `cd mcp-server && npm install`
3. Check env vars: `MEMORY_API_URL` (default: `http://localhost:8765`)

## Clearing All Data

```bash
# Clear all 3 stores via API
curl -X POST http://localhost:8765/api/v1/memories/delete_all \
  -H "Content-Type: application/json" \
  -d '{"user_id": "steven"}'

# Or manually:
# Qdrant: docker exec qdrant rm -rf /qdrant/storage/collections/openmemory
# Neo4j: docker exec neo4j cypher-shell -u neo4j -p openmemory "MATCH (n) DETACH DELETE n"
# SQLite: docker exec openmemory-mcp rm /usr/src/openmemory/db/openmemory.db
# Then restart: docker compose restart openmemory-mcp
```

## Performance

- **Slow adds**: Check if Ollama is swapping models. `OLLAMA_KEEP_ALIVE=30m` helps.
- **Slow search**: Normal is ~90-135ms. If >500ms, check Qdrant and Neo4j response times separately.
- **High VRAM**: With 3 models loaded (LLM ~2.7GB + embedding ~0.5GB + reserve), expect ~5GB usage on Ollama.
