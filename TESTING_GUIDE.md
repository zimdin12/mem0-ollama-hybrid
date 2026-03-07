# Memory System Testing Guide

Checklist for testing the memory extraction pipeline after any changes to prompts,
fact splitting, graph extraction, or dedup logic. Run after rebuilding the container.

## Prerequisites

```bash
docker compose build openmemory-mcp && docker compose up -d openmemory-mcp
```

Wait ~10s for the API to be ready, then verify: `curl -s http://localhost:8765/health`

---

## 1. Clear All Databases

Use the REST endpoint (clears Qdrant, Neo4j, and SQLite in one call):

```bash
curl -s -X POST http://localhost:8765/api/v1/memories/delete_all \
  -H "Content-Type: application/json" \
  -d '{"user_id":"steven"}'
```

Or manually:

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
| "User prefers..." instead of "Steven prefers..." | `conversation_memory` pipeline missing I/User replacement |
| Facts starting with "The" without subject | Dense chunk context injection not prepending topic |

---

## 7. LLM Model Evaluation & Fine-Tuning

This section covers how to evaluate new extraction LLMs, fine-tune sampling parameters,
and validate that a model switch doesn't regress quality.

### 7.1 Requirements for Extraction LLMs

The extraction pipeline needs models that can:
- Output **valid JSON** reliably (both fact lists and graph entity/relationship structures)
- Follow **system prompt instructions** precisely (not override with default behavior)
- Handle **structured extraction** without hallucinating extra entities or facts
- Work within **~4K output tokens** (extraction responses are short)

**Models that DON'T work** (tested and confirmed):
- **Thinking/reasoning models** (phi4-mini-reasoning, lfm2.5-thinking, exaone-deep, deepscaler) — emit thinking tokens that break JSON parsing, produce 5-10x more output than needed
- **Distilled models** (Crow-4B) — ignore system prompts entirely, generate unrelated content
- **Models without instruct tuning** — may not follow JSON format instructions

**Models that DO work** (tested):
- qwen3 family (4b instruct, 3.5:4b, 3.5:9b)
- ministral-3:3b
- gemma3:4b (works but has noise leaks)
- granite4:3b (works but has fact splitting issues)

### 7.2 Model Switching at Runtime

Switch the extraction LLM without restarting the container:

```bash
# Switch model + sampling config
curl -s -X PUT http://localhost:8765/api/v1/config/mem0/llm \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "ollama",
    "config": {
      "model": "NEW_MODEL_NAME",
      "ollama_base_url": "http://ollama:11434",
      "temperature": 0.3,
      "top_p": 0.8,
      "top_k": 20
    }
  }'
```

The API calls `reset_memory_client()` internally, which rebuilds the mem0 client with the new model. No container restart needed.

**Important**: The config API change is runtime-only. To persist across container restarts, also update `LLM_MODEL` in `docker-compose.yml`.

### 7.3 Per-Model Sampling Configs

Each model family has tuned sampling parameters in `openmemory/api/model_configs.py`.
The system auto-detects the family from the model name.

Current tuned configs:

| Family | Temp | top_p | top_k | PP | Notes |
|--------|------|-------|-------|----|-------|
| qwen3 | 0.7 | 0.8 | 20 | - | Reliable. Sensitive to low temps (lost 14 rels at t=0.15) |
| qwen3.5:4b | 0.15 | 0.8 | 20 | 0.5 | Best graph (184 edges). pp=0.5 sweet spot |
| qwen3.5 (9b+) | 0.1 | 0.8 | 20 | 1.5 | Original config was best. Config-insensitive |
| ministral | 0.05 | 0.8 | 40 | - | Best raw score. Very low temp works here |

To add a new model family:
1. Add an entry to `MODEL_CONFIGS` in `model_configs.py`
2. Set `LLM_MODEL` env var to the Ollama model name
3. Optionally set `LLM_FAMILY` env var if auto-detection doesn't match

### 7.4 Benchmark Scripts

All benchmarks run inside the container. Copy and execute:

```bash
docker cp SCRIPT.py openmemory-mcp:/usr/src/openmemory/ && \
MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/SCRIPT.py
```

| Script | Purpose | When to use |
|--------|---------|-------------|
| `benchmark_models.py` | Raw extraction quality across 9+ models. Tests prompts + filters directly via Ollama API (no mem0 pipeline). Measures facts, relationships, noise rejection, multi-person attribution. | Evaluating a **new model** you haven't tested before |
| `benchmark_finetune.py` | Parameter grid search. Tests multiple temp/top_k/top_p/presence_penalty combos per model. Scores each config on 11 standardized tests. | **Tuning sampling params** after selecting a model |
| `benchmark_system.py` | End-to-end pipeline test (13 phases). Tests the full API: insert -> dedup -> search -> graph -> pagination. Switches models via config API, clears all stores between models. | Validating a model works through the **full pipeline** |
| `benchmark_extended.py` | Deep quality inspection (62 inserts, 13 categories). Dumps all stored facts for manual review. Checks for first-person leaks, "User" refs, context-stripped facts, hallucinations. 15 search quality tests. | **Final validation** before switching production model |

### 7.5 Step-by-Step: Evaluating a New Model

#### Step 1: Pull the model
```bash
docker exec ollama ollama pull NEW_MODEL:TAG
```

#### Step 2: Raw extraction test
Run `benchmark_models.py` with the new model added to its `MODELS` list.
Look for:
- **Facts extracted**: Should be 40-55 from the standard test set (not 100+ which indicates noise)
- **Relationships**: Should be 20+ with correct source->target direction
- **Noise rejection**: Should score 5/5 (rejects irrelevant content)
- **Multi-person**: Should correctly attribute Jake to devops (tests entity disambiguation)
- **Speed**: Should complete 11 tests in under 60 seconds total

**Red flags** (immediate disqualification):
- 0 relationships (model can't do graph extraction)
- 100+ facts (model hallucinates or doesn't follow extraction rules)
- Noise score < 3/5 (model extracts from irrelevant content)
- Multi-person attribution wrong (model can't disambiguate entities)

#### Step 3: Parameter tuning
If the model passes Step 2, run `benchmark_finetune.py` with configs to test.
Start with the default (`temperature: 0.3, top_p: 0.8, top_k: 20`) and vary one param at a time:
- Temperature: try 0.05, 0.1, 0.15, 0.3, 0.5, 0.7
- top_k: try 20, 40, 64
- presence_penalty: try 0 (none), 0.5, 1.0, 1.5

Pick the config with the highest combined score (facts + relationships + noise + multi-person).

#### Step 4: Pipeline test
Add the model + best config to `benchmark_system.py`'s `MODELS` list and run it.
This tests the full API pipeline. Look for:
- **Memory count**: Should be 40-50 from the standard test set
- **Search**: 13/13 queries should return relevant results
- **Graph**: 50+ nodes, 100+ edges, diverse entity types
- **Self-refs**: 0 self-referential edges
- **Dedup**: 0 duplicate entries after re-insert
- **Speed**: Full run should complete in under 60 seconds

#### Step 5: Deep quality check
Run `benchmark_extended.py` for the final validation. Inspect the memory dump for:
- **First-person leaks**: Facts saying "I" or "my" instead of "Steven"
- **"User" references**: Facts saying "User prefers" instead of "Steven prefers"
- **Context-stripped facts**: Facts starting with "The" that lost their subject
- **Hallucinations**: Facts not present in the input text

#### Step 6: Switch production model
```bash
# Runtime switch
curl -s -X PUT http://localhost:8765/api/v1/config/mem0/llm \
  -H "Content-Type: application/json" \
  -d '{"provider":"ollama","config":{"model":"NEW_MODEL:TAG","ollama_base_url":"http://ollama:11434"}}'

# Persist in docker-compose.yml
# Change LLM_MODEL=NEW_MODEL:TAG in openmemory-mcp environment section

# Add config to model_configs.py
# Add entry with family name, patterns, and tuned options

# Pull model on startup (optional)
# Add pull command to ollama service's startup script in docker-compose.yml
```

### 7.6 Key Findings from Benchmarking

- **Prompt quality > sampling config**: Graph prompt fixes improved scores by +76 points. Config tuning improved scores by +5-15 points. Always fix prompts first.
- **All tested models produce 90%+ identical output** after prompt fixes. Differences are at the margins (1-2 facts, 5-10 graph edges).
- **Config sensitivity drops to near-zero** with good prompts. Bad prompts amplify config differences.
- **Speed is nearly identical** across 3-9B models on RTX 4090 (~0.4-0.9s per insert). Model size matters less than prompt complexity.
- **instruct variants matter for qwen3** (thinking mode breaks JSON). qwen3.5 doesn't have this issue.
- **Dense chunks are the hardest test**: Long texts with many entities stress context injection, fact splitting, and graph extraction simultaneously.

### 7.7 Clearing Ollama Context Between Models

When benchmarking multiple models, clear the previous model from VRAM:

```bash
# Unload current model from Ollama
curl -s -X POST http://ollama:11434/api/generate \
  -d '{"model":"CURRENT_MODEL","keep_alive":0}'

# Wait for VRAM to free
sleep 5

# Load new model (first request will trigger load)
```

The benchmark scripts handle this automatically, but it's important for manual testing too.
Ollama's `OLLAMA_MAX_LOADED_MODELS=3` allows keeping multiple models loaded if VRAM permits.
