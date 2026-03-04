"""
Comprehensive test of the memory system with Echoes of the Fallen game design document.
Tests insertion in various chunk sizes, validates all three storage layers,
and measures graph contribution to search quality.
"""
import json
import logging
import sys
import time
import uuid
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# --- Setup ---
sys.path.insert(0, '/usr/src/openmemory')
from app.database import SessionLocal
from app.utils.db import get_user_and_app
from app.utils.enhanced_memory import EnhancedMemoryManager
from app.models import Memory, MemoryState, MemoryStatusHistory

USER_ID = 'steven'
APP_ID = 'claude-code'

mgr = EnhancedMemoryManager()

def add_memory(text, label=""):
    """Add memory and sync to SQLite, return result."""
    t0 = time.time()
    result = mgr.smart_add_memory(text, USER_ID)
    elapsed = time.time() - t0

    # Sync to SQLite
    db = SessionLocal()
    try:
        user, app_obj = get_user_and_app(db, user_id=USER_ID, app_id=APP_ID)
        for mem in result.added_memories:
            mem_id = mem.get('id')
            content = mem.get('memory', '')
            if mem_id and content:
                try:
                    uid = uuid.UUID(mem_id)
                    existing = db.query(Memory).filter(Memory.id == uid).first()
                    if not existing:
                        db.add(Memory(id=uid, user_id=user.id, app_id=app_obj.id,
                                      content=content[:500], state=MemoryState.active))
                        db.add(MemoryStatusHistory(memory_id=uid, old_state=MemoryState.active,
                                                   new_state=MemoryState.active, changed_by=user.id))
                except Exception as e:
                    logging.warning(f"SQLite sync error: {e}")
        db.commit()
    finally:
        db.close()

    print(f"  [{label}] {result.status}: +{len(result.added_memories)} added, "
          f"-{len(result.skipped_facts)} skipped ({elapsed:.1f}s)")
    return result


def search_memory(query, limit=10):
    """Search and return results."""
    results = mgr.hybrid_search(query, USER_ID, limit=limit)
    return results


def count_qdrant():
    """Count vectors in Qdrant."""
    import requests
    r = requests.get('http://mem0_store:6333/collections/openmemory')
    data = r.json()
    return data.get('result', {}).get('points_count', 0)


def count_neo4j():
    """Count nodes and relationships in Neo4j."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "openmemory"))
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
    driver.close()
    return nodes, rels


def get_neo4j_entities():
    """Get all entities and relationships from Neo4j."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "openmemory"))
    entities = []
    relationships = []
    with driver.session() as session:
        for record in session.run("MATCH (n) RETURN n.name AS name, labels(n) AS labels ORDER BY n.name"):
            entities.append((record["name"], record["labels"]))
        for record in session.run("MATCH (n)-[r]->(m) RETURN n.name AS src, type(r) AS rel, m.name AS dst ORDER BY n.name"):
            relationships.append((record["src"], record["rel"], record["dst"]))
    driver.close()
    return entities, relationships


# ============================================================
# PHASE 1: Insert data in realistic chunks
# ============================================================
print("=" * 70)
print("PHASE 1: Inserting Echoes test data in varied chunk sizes")
print("=" * 70)

# Chunk 1: Personal intro (small, detail-poor)
chunk1 = "My name is steven and I have a game dev project called Echoes of the Fallen."
print("\nChunk 1: Personal intro (short)")
add_memory(chunk1, "intro")

# Chunk 2: Executive summary (big, detail-rich)
chunk2 = """Echoes of the Fallen is an ambitious voxel-based roguelike exploration game that combines permanent knowledge acquisition with temporary item systems in a beautiful but grim nature-reclaimed world. Actions generate persistent skills that "ripen" after death, creating a unique progression system where player knowledge and character abilities both advance through repeated playthroughs. The world becomes more dangerous with distance from spawn, encouraging gradual exploration and mastery."""
print("\nChunk 2: Executive summary (large, dense)")
add_memory(chunk2, "summary")

# Chunk 3: Gameplay loops (medium, structured)
chunk3 = """Primary Loop (15-45 minutes): Players spawn in a safe meadow area and venture into increasingly dangerous wilderness. Exploration activities generate nascent skills - swimming develops water affinity, falling repeatedly builds flight resistance, crafting creates tool mastery. When death occurs, items vanish but skills mature into permanent abilities, allowing deeper exploration in subsequent lives. Secondary Loop (2-4 hours): Over multiple deaths, players build a comprehensive skill portfolio while mentally mapping the world. Chest items become broken after death but can be salvaged for materials."""
print("\nChunk 3: Gameplay loops (medium)")
add_memory(chunk3, "loops")

# Chunk 4: Technical specs (heavy with versions and numbers)
chunk4 = """Primary Solution: Voxel Plugin 2.0 for UE5 (releasing early 2025, $349) with Nanite and Lumen support for modern rendering, built-in replication for future multiplayer, and graph-driven procedural generation. Interim Solution: Voxel Plugin Legacy (free) for prototyping and early development. Architecture Pattern: Dual Contouring for mesh generation preserving sharp edges for architectural elements. Chunk Management: 32x32x32 block chunks with aggressive LOD system, async processing on background threads, and memory pooling to reduce garbage collection impact."""
print("\nChunk 4: Technical specs (version-heavy)")
add_memory(chunk4, "tech")

# Chunk 5: Skill system design (game mechanics)
chunk5 = """Knowledge/Skill System: Action-Based Learning where swimming in rivers develops water magic affinity, repeated falling builds gliding abilities, and combat creates weapon masteries. Skill States are Fresh during gameplay and become Fully Acquired upon death. The Combination System allows multiple skills to create hybrid abilities like water plus earth equals mud manipulation. Implementation uses dual-layer progression with immediate run benefits plus long-term character growth, with thematic integration tying skills to isekai anime concepts of reincarnation with retained knowledge."""
print("\nChunk 5: Skill system (game design)")
add_memory(chunk5, "skills")

# Chunk 6: World generation (spatial concepts)
chunk6 = """World Generation uses Mixed Persistence with Handcrafted Core: static home area (spawn meadow, 1km radius) provides consistent safe zone, semi-persistent middle ring (5km radius) with important landmarks that shift but maintain themes, and fully regenerating outer wilderness with increasing danger and reward. Spatial Difficulty Scaling inspired by Valheim uses distance-based challenge rather than level-based progression, environmental mechanics increase complexity rather than just enemy stats, and clear visual cues signal danger levels to players. Technical implementation uses chunk-based generation with 512x512 meter regions."""
print("\nChunk 6: World generation (spatial)")
add_memory(chunk6, "world")

# Chunk 7: Art and audio (creative direction)
chunk7 = """Art Direction follows Beautiful Decay aesthetic - nature reclaiming civilization. Color palette uses desaturated earth tones with warm accent highlights. Dynamic day/night cycles with volumetric fog and atmospheric effects. Asset creation uses MagicaVoxel (free) for rapid prototyping, Blender for complex animations, and Qubicle ($189) for advanced features. Audio design focuses on nature soundscapes, procedural ambient tracks that respond to player actions, and diegetic sound cues tied to skill acquisition. Budget allocation recommends mid-tier $1000-2000 total."""
print("\nChunk 7: Art & audio direction")
add_memory(chunk7, "art")

# Chunk 8: Development phases (timeline)
chunk8 = """Development Phases: Phase 1 Foundation (Months 1-6) covers C++ fundamentals, UE5 learning, Voxel Plugin prototyping, vertical slice, and core systems with budget $700-1300. Phase 2 Content Development (Months 7-12) covers procedural biome generation, 3-5 distinct biomes, NPC systems, building/crafting, and skill combination polish with budget $500-900. Phase 3 Polish and Launch (Months 13-18) covers alpha testing, beta testing, marketing campaign, and platform submission with budget $800-1500. Total development time is 18 months solo with 6-month buffer."""
print("\nChunk 8: Development timeline")
add_memory(chunk8, "timeline")

# Chunk 9: Risk assessment (strategic)
chunk9 = """Risk Assessment: Technical risks include voxel performance optimization requiring sophisticated chunk management, skill system complexity with hundreds of potential combinations creating exponential design challenges, and memory management for voxel worlds. Scope Management: Research shows 70% of indie developers cite scope too large as major failure factor. Mitigation strategies include ruthless scope management, prototype early for risky systems, use proven tools like Voxel Plugin rather than custom engine, and establish strict performance budgets. Recommended moderate approach: 3-5 biomes, core skill system with 5-10 combinations, mixed world persistence."""
print("\nChunk 9: Risk assessment (strategic)")
add_memory(chunk9, "risks")

# Chunk 10: Market and success metrics (business)
chunk10 = """Market Positioning: Direct competitors include Bioprototype and other voxel roguelikes. Indirect competitors include Dead Cells, Hades, Subnautica, and Valheim. Unique Value Proposition: Only game combining voxel exploration with persistent skill-based progression and isekai-inspired death mechanics. Launch Goals: 7000+ Steam wishlists before early access, 80+ Metacritic score, $50000+ first-year revenue. Long-term: 100000+ lifetime sales, 500+ active Discord members. C++ learning path for PHP developers spans 4 phases over 12-16 weeks using Tom Looman's UE5 course."""
print("\nChunk 10: Market & success (business)")
add_memory(chunk10, "market")

# Chunk 11: Specific implementation details (technical)
chunk11 = """Swimming to Water Magic: Track time spent in water, depth of dives, swimming speed achieved. Generate Aquatic Affinity skill that unlocks water-walking, underwater breathing, water manipulation. Visual feedback: Character animations become more fluid in water as skill develops. Falling to Flight: Monitor fall damage taken, heights survived, creative fall solutions. Develop Aerial Grace leading to gliding, controlled falling, eventually limited flight. Implementation reduces fall damage progressively, adds gliding physics, creates updraft interactions."""
print("\nChunk 11: Implementation details (specific)")
add_memory(chunk11, "impl")

# Chunk 12: Duplicate test - send similar content again
chunk12 = "Echoes of the Fallen is a voxel-based roguelike exploration game. It uses a permanent skill system where skills ripen after death. The world uses distance-based difficulty scaling inspired by Valheim."
print("\nChunk 12: DUPLICATE TEST (similar to earlier)")
add_memory(chunk12, "dedup")

# ============================================================
# PHASE 2: Validate all storage layers
# ============================================================
print()
print("=" * 70)
print("PHASE 2: Validating storage layers")
print("=" * 70)

# Qdrant
qdrant_count = count_qdrant()
print(f"\nQdrant vectors: {qdrant_count}")

# Neo4j
neo4j_nodes, neo4j_rels = count_neo4j()
print(f"Neo4j nodes: {neo4j_nodes}, relationships: {neo4j_rels}")

# SQLite
db = SessionLocal()
sqlite_count = db.query(Memory).filter(Memory.state == MemoryState.active).count()
db.close()
print(f"SQLite memories: {sqlite_count}")

# ============================================================
# PHASE 3: Validate Neo4j entities and relationships
# ============================================================
print()
print("=" * 70)
print("PHASE 3: Neo4j graph quality")
print("=" * 70)

entities, relationships = get_neo4j_entities()
print(f"\nEntities ({len(entities)}):")
for name, labels in entities:
    label_str = ", ".join(l for l in labels if l != '__Entity__')
    print(f"  {name:40s} [{label_str}]")

print(f"\nRelationships ({len(relationships)}):")
for src, rel, dst in relationships:
    print(f"  {src:30s} --[{rel}]--> {dst}")

# Check for garbage entities
garbage_indicators = ['php', 'js', 'py', 'json', 'css', 'utilities', 'default', 'record']
garbage = [e for e in entities if e[0] in garbage_indicators]
print(f"\nGarbage entities (file extensions/generic words): {len(garbage)}")
for g in garbage:
    print(f"  BAD: {g[0]} [{g[1]}]")

# ============================================================
# PHASE 4: Search quality tests
# ============================================================
print()
print("=" * 70)
print("PHASE 4: Search quality tests")
print("=" * 70)

test_queries = [
    "What is Echoes of the Fallen?",
    "How does the skill system work?",
    "What technology is used for voxel rendering?",
    "What are the development phases and timeline?",
    "How does the world generation work?",
    "What are the main risks?",
    "What competitors exist?",
    "Who is Steven?",
    "swimming water magic abilities",
    "budget and cost estimates",
]

for query in test_queries:
    results = search_memory(query, limit=5)
    vector_results = [r for r in results if r.source == 'vector']
    graph_results = [r for r in results if r.source == 'graph']
    temporal_results = [r for r in results if r.source == 'temporal']
    print(f"\n--- Query: \"{query}\" ---")
    print(f"  Total: {len(results)} (vector:{len(vector_results)}, graph:{len(graph_results)}, temporal:{len(temporal_results)})")
    for r in results[:3]:
        print(f"  [{r.source:8s}] {r.score:.3f} | {r.content[:80]}")

# ============================================================
# PHASE 5: Dedup quality check
# ============================================================
print()
print("=" * 70)
print("PHASE 5: Dedup quality (check for duplicates in Qdrant)")
print("=" * 70)

import requests
r = requests.post('http://mem0_store:6333/collections/openmemory/points/scroll',
                   json={"limit": 200, "with_payload": True})
points = r.json().get('result', {}).get('points', [])

# Find near-duplicates by content
from collections import Counter
contents = [p.get('payload', {}).get('data', '') for p in points]
content_counts = Counter(contents)
dupes = {c: n for c, n in content_counts.items() if n > 1}
print(f"\nTotal vectors: {len(points)}")
print(f"Exact duplicates: {len(dupes)}")
for content, count in dupes.items():
    print(f"  [{count}x] {content[:70]}")

# Check for fragment-like entries
fragments = [c for c in contents if len(c) < 20 or
             any(c.lower().startswith(ext) for ext in ['php', 'js', 'py', 'json', 'css', '3 with', '2+)'])]
print(f"\nBroken fragments: {len(fragments)}")
for f in fragments:
    print(f"  FRAGMENT: {f[:60]}")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
