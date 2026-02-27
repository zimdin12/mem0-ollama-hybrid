"""
Custom prompts optimized for qwen3:4b-instruct (small 4B model).

Includes:
1. FACT_EXTRACTION_PROMPT - Extracts facts from dense prose documents
2. UPDATE_MEMORY_PROMPT - Smart deduplication with bias toward preserving information
3. GRAPH_ENTITY_RELATIONSHIP_PROMPT - Extracts rich graph entities and relationships

Key improvements:
- Handles prose paragraphs better than default mem0 prompts
- Clear, objective rules (not subjective judgments)
- Preserves distinct facts (less aggressive deduplication)
- Extracts comprehensive entity relationships for graph database
"""

from datetime import datetime

QWEN3_FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer. Extract ALL distinct facts from conversations.

YOUR TASK:
Read the input text carefully and extract EVERY distinct piece of information as a separate fact.
Each fact should be:
- A complete, standalone statement
- Specific and factual
- About a person, preference, plan, activity, or detail

EXTRACTION RULES:
1. Extract EVERY piece of information, even if related to same topic
2. Break down complex sentences into multiple simple facts
3. Keep each fact focused on ONE specific detail
4. Do NOT combine related facts - keep them separate
5. Do NOT skip facts because they seem minor - extract everything

EXAMPLES:

Input: "Hi, I am looking for a restaurant"
Output: {{"facts": ["Looking for a restaurant"]}}

Input: "My name is Steven and I have a game development project"
Output: {{"facts": ["Name is Steven", "Has a game development project"]}}

Input: "Echoes of the Fallen is a voxel-based roguelike exploration game that combines permanent knowledge with temporary items in a grim nature-reclaimed world"
Output: {{
  "facts": [
    "Game is called Echoes of the Fallen",
    "Echoes of the Fallen is voxel-based",
    "Echoes of the Fallen is a roguelike",
    "Echoes of the Fallen is an exploration game",
    "Game combines permanent knowledge acquisition",
    "Game has temporary item systems",
    "Game world is grim and nature-reclaimed"
  ]
}}

IMPORTANT:
- Today's date: {datetime.now().strftime("%Y-%m-%d")}
- Return ONLY JSON format: {{"facts": ["fact1", "fact2", ...]}}
- Empty list if no facts: {{"facts": []}}
- Do NOT skip trivial-seeming details
- Extract facts from user AND assistant messages
- Ignore system messages
- If input is dense prose, break into MANY small facts

You will now receive the input. Extract ALL facts as JSON:
"""

QWEN3_GRAPH_RELATIONSHIP_PROMPT = """
Extract ALL meaningful relationships from the text to build a comprehensive knowledge graph.

EXTRACTION RULES:
1. Extract EVERY entity mentioned (people, projects, concepts, places, objects, skills, systems)
2. Create relationships for ALL connections between entities, not just obvious ones
3. Be thorough - extract relationships even if they seem minor
4. Break down complex descriptions into multiple simple relationships

RELATIONSHIP TYPES - Use clear, simple verbs:
- Ownership: "has", "owns", "possesses"
- Creation: "creates", "develops", "builds", "produces"
- Attributes: "is", "features", "includes"
- Actions: "does", "performs", "executes"
- Associations: "works_on", "relates_to", "involves"
- Skills: "develops_skill", "improves", "learns"
- Properties: "is_type", "has_property", "characterized_by"

IMPORTANT:
- Extract relationships between ALL entities mentioned in the list
- Do NOT skip relationships because they seem obvious or minor
- Each relationship should be simple and clear
- Create multiple relationships for entities with multiple connections
- Use the exact entity names provided in the entity list
"""

QWEN3_OPTIMIZED_UPDATE_PROMPT = """You are a smart memory manager. You control memory storage with four operations:
(1) ADD - add new fact to memory
(2) UPDATE - merge or enhance existing memory
(3) DELETE - remove contradicted memory
(4) NONE - no change needed

DECISION RULES (follow these exactly):

1. **ADD** - Use when:
   - The new fact provides NEW information not in memory
   - The new fact is about a DIFFERENT topic/aspect than existing memories
   - When unsure if it's duplicate, ADD IT (better to have detail than miss information)

2. **UPDATE** - Use ONLY when:
   - The new fact is clearly BETTER/MORE DETAILED version of existing fact
   - The new fact ENHANCES existing memory with additional details

3. **DELETE** - Use ONLY when:
   - The new fact DIRECTLY CONTRADICTS existing memory

4. **NONE** - Use ONLY when:
   - The new fact is EXACTLY THE SAME as existing memory
   - NOT just similar - must be nearly identical

IMPORTANT GUIDELINES FOR SMALL FACTS:
- Different facts about the same topic should be SEPARATE memories (ADD each)
- Each discrete piece of information should be its own memory entry
- Only combine facts when one truly enhances the other, not when they're just related

CRITICAL RULES:
- When in doubt, ADD the fact (preserve information)
- Keep related facts SEPARATE unless one truly enhances the other
- Only UPDATE when the new fact makes the old fact obsolete
- Only DELETE for direct contradictions
- Generate new sequential IDs for ADD operations (next available number)
- Return ONLY valid JSON, no other text
"""


def get_qwen3_fact_extraction_prompt():
    """Returns the optimized fact extraction prompt for qwen3:4b"""
    return QWEN3_FACT_EXTRACTION_PROMPT


def get_qwen3_update_prompt():
    """Returns the optimized update memory prompt for qwen3:4b"""
    return QWEN3_OPTIMIZED_UPDATE_PROMPT


def get_qwen3_graph_relationship_prompt():
    """Returns the optimized graph relationship extraction prompt for qwen3:4b"""
    return QWEN3_GRAPH_RELATIONSHIP_PROMPT
