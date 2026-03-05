"""
JSON-based graph extraction for qwen3:4b (replaces unreliable tool calling).

Problem: qwen3:4b cannot reliably use Ollama's tool/function calling format.
Entity extraction returns empty tool_calls, relationship extraction fails.

Solution: Use Ollama's `format: 'json'` mode with a single prompt that extracts
both entities and relationships in one call. This is:
- More reliable (JSON mode forces valid JSON output)
- Faster (1 LLM call instead of 3-4)
- Higher quality (LLM sees full context for both entities and relationships)
"""

import json
import re
import logging
from mem0.memory.graph_memory import MemoryGraph

logger = logging.getLogger(__name__)

# Save originals for potential restore
_original_add = MemoryGraph.add
_original_retrieve_nodes = MemoryGraph._retrieve_nodes_from_data
_original_establish_relations = MemoryGraph._establish_nodes_relations_from_data

# Entities that should NOT be in the graph (file extensions, generic words)
_ENTITY_BLOCKLIST = frozenset([
    'php', 'js', 'jsx', 'ts', 'tsx', 'py', 'json', 'yaml', 'yml', 'css', 'html',
    'xml', 'md', 'txt', 'sh', 'bat', 'sql', 'csv', 'log', 'conf', 'cfg', 'ini',
    'utilities', 'default', 'record', 'entity', 'unknown', 'other', 'none', 'null',
    'true', 'false', 'yes', 'no', 'n/a', 'etc', 'i.e', 'e.g',
])

GRAPH_EXTRACTION_PROMPT = """Extract entities and relationships from text. Return ONLY a JSON object.

Output format:
{"entities": [{"name": "...", "type": "..."}, ...], "relationships": [{"source": "...", "relation": "...", "target": "..."}, ...]}

ENTITY TYPE RULES (use the MOST SPECIFIC type — "concept" is LAST RESORT):
- "person": people, users, developers (steven, alice)
- "project": games, applications, products being built (echoes of the fallen)
- "technology": languages, engines, protocols (c++, unreal engine 5, http)
- "framework": specific frameworks (laravel, react, godot)
- "tool": software tools (magicavoxel, blender, vs code, voxel plugin)
- "game_element": in-game mechanics, abilities, resources, systems (water magic, skill acquisition, chunk management, dual contouring, nanite)
- "metric": numbers, targets, budgets, timelines (18 months, $50000 revenue, 80+ metacritic)
- "phase": development phases, milestones (pre-production, alpha testing, phase 1)
- "place": locations, real or in-game areas (home meadow, wilderness, exploration ring)
- "organization": companies, teams, communities (indie developers, discord community)
- "concept": ONLY for abstract ideas that fit NO other type above

RELATIONSHIP RULES:
- Connect entities to their PARENT: features/phases/metrics belong to their project
- Use the person/user as a hub: "steven" develops projects, has skills, uses tools
- Game elements connect to their project, NOT to each other randomly
- Phases connect to their project: project has_phase phase
- Metrics connect to their project: project has_target metric
- DO NOT connect two entities that mean the same thing (e.g., "water" and "water magic")
- DO NOT create circular relationships (A->B->A)

RELATION VERBS:
- develops, creates, builds (person -> project)
- is_a, has_role (person -> role/skill)
- uses, prefers, learning (person -> technology/tool)
- features, contains, has_phase (project -> element/phase)
- has_target, has_budget (project -> metric)
- part_of, belongs_to (element -> project)
- built_with, uses (project -> technology)
- inspired_by (project -> project)
- unlocks, enables (game_element -> game_element, within same project)

EXAMPLE:
Text: "Steven is a PHP developer building Echoes of the Fallen, a voxel roguelike. Total development time is 18 months. The game uses C++ and dual contouring for terrain. Phase 1 covers foundation work months 1-6."
Output:
{"entities": [
  {"name": "steven", "type": "person"},
  {"name": "echoes of the fallen", "type": "project"},
  {"name": "php", "type": "technology"},
  {"name": "c++", "type": "technology"},
  {"name": "dual contouring", "type": "game_element"},
  {"name": "18 months development", "type": "metric"},
  {"name": "phase 1 foundation", "type": "phase"}
], "relationships": [
  {"source": "steven", "relation": "is_a", "target": "php developer"},
  {"source": "steven", "relation": "develops", "target": "echoes of the fallen"},
  {"source": "echoes of the fallen", "relation": "built_with", "target": "c++"},
  {"source": "echoes of the fallen", "relation": "features", "target": "dual contouring"},
  {"source": "echoes of the fallen", "relation": "has_target", "target": "18 months development"},
  {"source": "echoes of the fallen", "relation": "has_phase", "target": "phase 1 foundation"}
]}

Rules:
- Keep entity names concise (2-4 words max), lowercase
- Do NOT create entities from file extensions (.php, .js)
- Do NOT create entities from generic words (utilities, default, record)
- For self-references (I, me, my), use "USER_ID" as the entity name"""


def _json_extract_graph(self, data, filters):
    """
    Extract entities and relationships using JSON mode (single LLM call).
    Returns (entity_type_map, relationships).
    """
    user_id = filters.get('user_id', 'user')
    prompt = GRAPH_EXTRACTION_PROMPT.replace("USER_ID", user_id)

    try:
        # Use Ollama's JSON mode directly
        import requests
        import os

        ollama_url = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
        model = os.environ.get('LLM_MODEL', 'qwen3:4b-instruct-2507-q4_K_M')

        resp = requests.post(f'{ollama_url}/api/chat', json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': data},
            ],
            'stream': False,
            'format': 'json',
        }, timeout=120)

        if resp.status_code != 200:
            logger.warning(f"Graph JSON extraction failed: HTTP {resp.status_code}")
            return {}, []

        raw_resp = resp.json()
        if not isinstance(raw_resp, dict):
            logger.warning(f"Graph JSON extraction: unexpected response type {type(raw_resp)}")
            return {}, []
        msg = raw_resp.get('message', {})
        if isinstance(msg, str):
            content = msg
        elif isinstance(msg, dict):
            content = msg.get('content', '')
        else:
            content = ''
        if not content:
            logger.warning("Graph JSON extraction returned empty content")
            return {}, []

        parsed = json.loads(content)

        # Build entity_type_map
        entity_type_map = {}
        raw_entities = parsed.get('entities', [])
        for ent in raw_entities:
            if isinstance(ent, str):
                name = ent.strip()
                etype = 'concept'
            elif isinstance(ent, dict):
                name = ent.get('name', '').strip()
                etype = ent.get('type', 'concept').strip()
            else:
                continue
            if not name:
                continue
            # Normalize
            name_key = name.lower().replace(' ', '_')
            etype_key = etype.lower().replace(' ', '_')
            # Filter blocked entities
            if name_key in _ENTITY_BLOCKLIST:
                continue
            if len(name_key) < 2:
                continue
            entity_type_map[name_key] = etype_key

        # Build relationships list
        relationships = []
        raw_rels = parsed.get('relationships', [])
        for rel in raw_rels:
            if not isinstance(rel, dict):
                continue
            source = str(rel.get('source', '')).strip().lower().replace(' ', '_')
            target = str(rel.get('target', '')).strip().lower().replace(' ', '_')
            relation = str(rel.get('relation', '')).strip().lower().replace(' ', '_')

            if not source or not target or not relation:
                continue
            # Skip self-references (exact match)
            if source == target:
                continue
            # Skip fuzzy self-references (one contains the other)
            if len(source) > 2 and len(target) > 2 and (source in target or target in source):
                logger.info(f"Skipping fuzzy self-ref: {source} -> {target}")
                continue
            # Skip blocked entities
            if source in _ENTITY_BLOCKLIST or target in _ENTITY_BLOCKLIST:
                continue
            # Ensure both entities are in the map (add if missing)
            if source not in entity_type_map:
                entity_type_map[source] = 'concept'
            if target not in entity_type_map:
                entity_type_map[target] = 'concept'

            relationships.append({
                'source': source,
                'relationship': relation,
                'destination': target,
            })

        logger.info(f"Graph JSON extraction: {len(entity_type_map)} entities, {len(relationships)} relationships")
        return entity_type_map, relationships

    except json.JSONDecodeError as e:
        logger.warning(f"Graph JSON parse error: {e}")
        return {}, []
    except Exception as e:
        logger.warning(f"Graph JSON extraction error: {e}")
        return {}, []


def _patched_add(self, data, filters):
    """
    Replacement for MemoryGraph.add() that uses JSON extraction.
    Skips the unreliable tool-calling-based pipeline entirely.
    """
    # Step 1: Extract entities and relationships in one JSON call
    entity_type_map, to_be_added = _json_extract_graph(self, data, filters)

    if not entity_type_map and not to_be_added:
        logger.info("Graph extraction returned nothing, skipping graph update")
        return {"deleted_entities": [], "added_entities": []}

    # Step 2: Search for existing similar nodes (uses embeddings, no LLM)
    search_output = self._search_graph_db(
        node_list=list(entity_type_map.keys()), filters=filters
    )

    # Step 3: Skip deletion step (unreliable with small LLMs, and risky)
    # The old pipeline used an LLM call to decide what to delete — too error-prone.
    # We only add/update, never delete from graph via extraction.

    # Step 4: Add entities and relationships to Neo4j
    added_entities = self._add_entities(to_be_added, filters, entity_type_map)

    return {"deleted_entities": [], "added_entities": added_entities}


def _patched_retrieve_nodes(self, data, filters):
    """Stub — extraction happens in _patched_add via JSON."""
    return {}


def _patched_establish_relations(self, data, filters, entity_type_map):
    """Stub — extraction happens in _patched_add via JSON."""
    return []


def apply_patch():
    """Apply JSON-based graph extraction patch for qwen3:4b."""
    MemoryGraph.add = _patched_add
    MemoryGraph._retrieve_nodes_from_data = _patched_retrieve_nodes
    MemoryGraph._establish_nodes_relations_from_data = _patched_establish_relations
    print("✓ Applied qwen3:4b entity and relationship extraction fixes for graph extraction")


def remove_patch():
    """Remove the patch and restore original behavior."""
    MemoryGraph.add = _original_add
    MemoryGraph._retrieve_nodes_from_data = _original_retrieve_nodes
    MemoryGraph._establish_nodes_relations_from_data = _original_establish_relations
    print("✓ Removed qwen3:4b graph extraction patch")
