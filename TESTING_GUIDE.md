# Memory System Testing Guide

Checklist for testing the memory extraction pipeline after any changes to prompts,
fact splitting, graph extraction, or dedup logic. Run after rebuilding the container.

## Prerequisites

```bash
docker compose build openmemory-mcp && docker compose up -d openmemory-mcp
```

Wait ~10s for the API to be ready, then verify: `curl -s http://localhost:8765/health`

## 1. Clear All Databases

```bash
# Qdrant — delete and recreate collection
curl -s -X DELETE "http://localhost:6333/collections/openmemory"
curl -s -X PUT "http://localhost:6333/collections/openmemory" \
  -H "Content-Type: application/json" \
  -d '{"vectors":{"size":1024,"distance":"Cosine"}}'

# Neo4j — delete all nodes and relationships
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n) DETACH DELETE n'

# SQLite — delete all memory records
docker exec openmemory-mcp python -c "
from app.database import SessionLocal
from app.models import Memory
db = SessionLocal()
db.query(Memory).delete()
db.commit()
db.close()
print('SQLite cleared')
"
```

## 2. Test Data Insertion

Insert test data in stages. After each stage, verify all 3 databases.

### Stage A: Tiny chunk (1 fact)
```bash
curl -s -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Steven prefers dark mode in all his editors","user_id":"steven"}'
```
**Expected**: 1 vector in Qdrant, 1 SQLite entry, graph nodes for "steven" and "dark mode"

### Stage B: Small chunk (2-3 facts)
```bash
curl -s -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Steven uses Docker on Windows with WSL2. His GPU is an RTX 4090 with 24GB VRAM.","user_id":"steven"}'
```
**Expected**: 2-3 vectors, "steven" connected to "docker", "rtx 4090" in graph

### Stage C: Medium chunk with named project
```bash
curl -s -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Echoes of the Fallen is a voxel-based roguelike game. Total development time is 18 months solo. The budget is $1000-2000. Phase 1 covers months 1-6 for foundation work. A 6-month buffer is built into estimates.","user_id":"steven"}'
```
**Expected**:
- Facts should contain "Echoes of the Fallen" context (not orphaned "Total development time 18 months solo")
- Graph: `echoes_of_the_fallen` entity connected to phases, budget, timeline
- Entity types should include `project`, `phase`, `metric` — not all `concept`

### Stage D: Large dense chunk
Use the full Echoes test data file or a large section of it.

### Stage E: Duplicate test
Re-insert Stage C content. **Expected**: 0 new facts added, skipped_duplicates > 0.

### Stage F: Sparse/vague content
```bash
curl -s -X POST http://localhost:8765/api/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"text":"Players can do things in the world and get better over time","user_id":"steven"}'
```
**Expected**: May extract 1 fact or none (too vague). Should NOT create garbage graph nodes.

## 3. Database Verification Queries

### Qdrant (vector count)
```bash
curl -s "http://localhost:6333/collections/openmemory" | python -m json.tool | grep points_count
```

### Neo4j (nodes and relationships)
```bash
# All nodes with types
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n) WHERE n.user_id = "steven" RETURN n.name, labels(n) ORDER BY n.name'

# All relationships
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n)-[r]->(m) WHERE n.user_id = "steven" RETURN n.name, type(r), m.name ORDER BY n.name'

# Self-referential edges (MUST return 0 rows)
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n)-[r]->(n) RETURN n.name, type(r)'
```

### SQLite (memory count)
```bash
docker exec openmemory-mcp python -c "
from app.database import SessionLocal
from app.models import Memory, MemoryState
db = SessionLocal()
total = db.query(Memory).count()
active = db.query(Memory).filter(Memory.state != 'deleted').count()
print(f'Total: {total}, Active: {active}')
db.close()
"
```

## 4. Data Consistency Checks

### Check 1: Self-contained facts
Search for generic terms and verify results include their parent context:
```bash
# Should mention "Echoes of the Fallen" in results
curl -s -X POST http://localhost:8765/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"development time","user_id":"steven","limit":5}' | python -m json.tool

# Should mention project context
curl -s -X POST http://localhost:8765/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"budget estimates buffer","user_id":"steven","limit":5}' | python -m json.tool
```

### Check 2: Entity type diversity
Count entity types in Neo4j. "concept" should NOT be >50% of all nodes:
```bash
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n) WHERE n.user_id = "steven" UNWIND labels(n) AS label RETURN label, count(*) ORDER BY count(*) DESC'
```

### Check 3: Hub entity connectivity
Key entities should have multiple connections:
```bash
# Steven should connect to project, tools, preferences
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n {name: "steven"})-[r]->(m) RETURN type(r), m.name ORDER BY type(r)'

# Project should connect to features, phases, technologies
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n {name: "echoes_of_the_fallen"})-[r]->(m) RETURN type(r), m.name ORDER BY type(r)'
```

### Check 4: No self-referential edges
```bash
docker exec neo4j cypher-shell -u neo4j -p openmemory \
  'MATCH (n)-[r]->(n) RETURN n.name, type(r)'
```
**MUST return 0 rows.** If any exist, the self-ref filter is broken.

### Check 5: Search returns multiple sources
```bash
curl -s -X POST http://localhost:8765/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"steven","user_id":"steven","limit":10}' | python -m json.tool
```
Check `sources_used` — should include at least `vector` and `graph` (temporal depends on recency).

### Check 6: Dedup works
Insert the same content twice. Second insert should return `skipped_duplicates > 0` and add 0 new facts.

## 5. MCP Tool Testing

Test via the MCP SSE endpoint (if connected) or curl the MCP server directly:

```bash
# search_memory
# (use MCP client or Claude Code with openmemory MCP configured)

# add_memories — verify dedup works through MCP path too
# delete_memories — verify returns deleted content + related memories
# get_related_memories — verify returns grouped results by source
```

## 6. Common Failure Patterns

| Symptom | Likely Cause |
|---------|-------------|
| Facts like "Total development time 18 months" without project name | `_inject_context()` not working or topic detection failed |
| 90%+ nodes typed as "concept" | Graph extraction prompt not specific enough about entity types |
| Self-referential edges (A->A) | Fuzzy self-ref filter or element-ID guard missing |
| Hub entities (steven, project) with 0-1 connections | Prompt not instructing relationship hierarchy |
| Duplicate facts after re-insert | Dedup threshold too high (>0.85) or embedding model changed |
| Graph empty after insert | Graph extraction running async — wait 5-10s and recheck |
| Search returns only vector results | Neo4j down, or graph search entity extraction failing |
