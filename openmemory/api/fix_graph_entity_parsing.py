"""
Monkey-patch for mem0's graph entity extraction to fix qwen3:4b output parsing.

Problem: qwen3:4b includes helpful parenthetical notes in entity extraction:
  {"entity": "Steven (source entity", "entity_type": "steven_debug_test)"}

This breaks the entity names when mem0 processes them. This patch cleans
the entity names by removing everything after "(" in both entity and entity_type.
"""

import re
import logging
from mem0.memory.graph_memory import MemoryGraph
from mem0.graphs.tools import EXTRACT_ENTITIES_TOOL, EXTRACT_ENTITIES_STRUCT_TOOL

logger = logging.getLogger(__name__)

# Save original method
_original_retrieve_nodes = MemoryGraph._retrieve_nodes_from_data


def _patched_retrieve_nodes(self, data, filters):
    """
    Patched version that cleans qwen3:4b's parenthetical notes from entities.
    """
    # Use mem0's tools
    _tools = [EXTRACT_ENTITIES_TOOL]
    if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
        _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]

    # Enhanced prompt for qwen3:4b - forces tool calling with clear structure
    system_prompt = f"""You are a precise entity extraction system. Your task is to identify ALL entities and their types from the provided text.

CRITICAL INSTRUCTIONS:
1. You MUST use the extract_entities function call - do NOT write explanatory text
2. Extract EVERY entity mentioned in the text, no matter how many
3. For self-references ('I', 'me', 'my'), use '{filters['user_id']}' as the entity name
4. Keep entity names concise - use the actual name, not descriptions
5. Choose specific entity types: person, organization, place, product, concept, skill, etc.

FORMAT: Use the extract_entities function with an array of {{"entity": "name", "entity_type": "type"}} objects.

DO NOT provide explanations or descriptions - only call the function."""

    search_results = self.llm.generate_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract all entities from this text:\n\n{data}"},
        ],
        tools=_tools,
    )

    entity_type_map = {}

    try:
        for tool_call in search_results.get("tool_calls", []):
            if tool_call["name"] != "extract_entities":
                continue
            for item in tool_call["arguments"]["entities"]:
                # Handle mixed formats: qwen3:4b sometimes returns strings instead of dicts
                if isinstance(item, str):
                    entity = item.strip()
                    entity_type = 'entity'
                elif isinstance(item, dict):
                    entity_raw = item.get("entity", "")
                    entity_type_raw = item.get("entity_type", "entity")

                    # Clean entity name: remove parentheses, arrows, and "entity type" suffix
                    entity = re.sub(r'\s*\(.*', '', entity_raw).strip()
                    entity = re.sub(r'\s*→.*', '', entity).strip()

                    # Clean entity_type: remove parentheses, arrows, and quotes
                    entity_type = re.sub(r'\).*', '', entity_type_raw)
                    entity_type = re.sub(r'\s*\(.*', '', entity_type)
                    entity_type = re.sub(r'\s*→.*', '', entity_type)
                    entity_type = entity_type.strip().strip('"\'')

                    # Additional cleaning: take only the base type before comma or "due"
                    if ',' in entity_type:
                        entity_type = entity_type.split(',')[0].strip()
                    if '_due_to' in entity_type:
                        entity_type = entity_type.split('_due_to')[0].strip()
                    if ' due' in entity_type:
                        entity_type = entity_type.split(' due')[0].strip()

                    # If entity_type looks like a user_id, default to reasonable type
                    if len(entity_type) > 20 or '_test' in entity_type or entity_type == entity.replace(' ', '_'):
                        entity_type = 'entity'
                else:
                    continue

                if entity and entity_type:
                    entity_type_map[entity] = entity_type

    except Exception as e:
        logger.exception(
            f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
        )

    # Apply mem0's standard cleanup
    entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}

    logger.debug(f"Entity type map (after qwen3 cleaning): {entity_type_map}\n search_results={search_results}")

    return entity_type_map


def _patched_establish_relations(self, data, filters, entity_type_map):
    """
    Patched version with improved prompt for qwen3:4b relationship extraction.
    """
    from mem0.graphs.tools import RELATIONS_TOOL, RELATIONS_STRUCT_TOOL

    user_identity = f"user_id: {filters['user_id']}"
    if filters.get("agent_id"):
        user_identity += f", agent_id: {filters['agent_id']}"
    if filters.get("run_id"):
        user_identity += f", run_id: {filters['run_id']}"

    system_prompt = f"""You are a precise relationship extraction system. Your task is to identify ALL relationships between the provided entities.

CRITICAL INSTRUCTIONS:
1. You MUST use the establish_relationships function call - do NOT write explanatory text
2. Extract EVERY relationship you can find between the entities
3. For self-references ('I', 'me', 'my'), use '{user_identity}' as the source entity
4. Use concise relationship names: has, is_a, works_on, located_in, part_of, etc.
5. Only create relationships between entities that are actually connected in the text

FORMAT: Use the establish_relationships function with an array of {{"source": "entity1", "relationship": "rel_type", "destination": "entity2"}} objects.

DO NOT provide explanations or descriptions - only call the function with the relationships."""

    entity_list = list(entity_type_map.keys())
    user_content = f"Entities to connect: {entity_list}\n\nText: {data}"

    _tools = [RELATIONS_TOOL]
    if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
        _tools = [RELATIONS_STRUCT_TOOL]

    extracted_entities = self.llm.generate_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=_tools,
    )

    entities = []
    if extracted_entities.get("tool_calls"):
        entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

    entities = self._remove_spaces_from_entities(entities)
    logger.debug(f"Extracted entities: {entities}")
    return entities


def apply_patch():
    """Apply the monkey patch to fix qwen3:4b entity parsing."""
    MemoryGraph._retrieve_nodes_from_data = _patched_retrieve_nodes
    MemoryGraph._establish_nodes_relations_from_data = _patched_establish_relations
    print("✓ Applied qwen3:4b entity and relationship extraction fixes for graph extraction")


def remove_patch():
    """Remove the monkey patch and restore original behavior."""
    MemoryGraph._retrieve_nodes_from_data = _original_retrieve_nodes
    print("✓ Removed qwen3:4b entity parsing patch")
