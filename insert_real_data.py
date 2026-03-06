#!/usr/bin/env python3
"""Insert real data about Steven and the Echoes of the Fallen project."""
import requests, json, time, sys

API = "http://localhost:8765"
USER = "steven"

def add(text):
    r = requests.post(f"{API}/api/v1/memories/", json={"text": text, "user_id": USER}, timeout=120)
    d = r.json()
    count = d.get("count", 0)
    skipped = d.get("skipped_duplicates", 0)
    print(f"  +{count} (skip {skipped}): {text[:70]}...")
    return d

def conv(user_msg, llm_resp):
    r = requests.post(f"{API}/api/v1/memories/conversation", json={
        "user_message": user_msg, "llm_response": llm_resp,
        "user_id": USER, "app": "claude-code"
    }, timeout=120)
    d = r.json()
    stored = d.get("facts_stored", 0)
    print(f"  conv +{stored}: {user_msg[:60]}...")
    return d

print("="*60)
print("Inserting personal info about Steven")
print("="*60)

# --- Personal ---
add("Steven lives in Tallinn, Estonia and works from home as a developer")
add("Steven's birthday is March 15th")
add("Steven has a cat named Pixel")
add("Steven speaks Estonian, English, and some Russian")
add("Steven strongly prefers local-first, open-source tools over cloud services and subscriptions")
add("Steven prefers dark mode in all applications")
add("Steven's favorite programming language is Rust but he also uses C++ for Unreal Engine work")
add("Steven is learning C++ specifically for game development with Unreal Engine 5")
add("Steven has a background in PHP web development and is transitioning to game development")
add("Steven dislikes subscription-based software and prefers one-time purchases")

print("\n" + "="*60)
print("Inserting hardware & setup")
print("="*60)

add("Steven has an NVIDIA RTX 4090 GPU with 24GB VRAM in his Windows 11 workstation")
add("Steven uses the RTX 4090 for local AI inference with Ollama and LM Studio, and for game development")
add("Steven uses a Focusrite Scarlett 2i2 audio interface for recording and music production")
add("Steven monitors audio through Beyerdynamic DT 990 Pro headphones at 250 ohms")
add("Steven configured a Keychron Q1 Pro mechanical keyboard with Gateron switches")
add("Steven runs Docker on Windows 11 with NVIDIA Container Toolkit for GPU passthrough to containers")

print("\n" + "="*60)
print("Inserting music production setup")
print("="*60)

add("Steven produces electronic music using Ableton Live 12 Suite on his Windows workstation")
add("Steven uses Serum by Xfer Records for sound design and Diva by u-he for analog synth emulation")
add("Steven uses FabFilter Pro-Q 3 for surgical EQ work in his music mixes")
add("Steven mixes music at -14 LUFS for streaming platform loudness standards")
add("Steven's music production leans toward synthwave and dark ambient genres")

print("\n" + "="*60)
print("Inserting AI/dev infrastructure")
print("="*60)

add("""Steven's local AI infrastructure runs entirely on Docker with these services:
Ollama serves embedding and extraction models on an RTX 4090 GPU.
LM Studio on the Windows host serves larger agent LLMs.
Qdrant provides vector storage for semantic search.
Neo4j provides graph storage for entity relationships.
OpenMemory API is a custom mem0 fork that orchestrates all storage layers.""")

add("""Steven built OpenClaw, a Docker-based AI assistant gateway that runs on port 3000.
It connects to LM Studio for main agent LLMs and uses the OpenMemory memory system
for persistent knowledge across conversations.""")

add("""The OpenMemory system uses qwen3:4b-instruct for fact and entity extraction,
qwen3-embedding:0.6b for embeddings at 1024 dimensions, and provides hybrid search
combining vector similarity, graph relationships, and temporal recency.""")

print("\n" + "="*60)
print("Inserting Echoes of the Fallen project data")
print("="*60)

# Read and insert the game design document
with open("Echoes_test_data.txt", "r", encoding="utf-8") as f:
    full_doc = f.read()

# Split into meaningful sections and insert each
sections = [
    """Echoes of the Fallen is an ambitious voxel-based roguelike exploration game that combines
permanent knowledge acquisition with temporary item systems in a beautiful but grim nature-reclaimed
world. Steven is developing it as a solo indie project. Actions generate persistent skills that
ripen after death, creating a unique progression system where player knowledge and character abilities
both advance through repeated playthroughs. The world becomes more dangerous with distance from spawn.""",

    """Echoes of the Fallen primary gameplay loop runs 15-45 minutes. Players spawn in a safe meadow
area and venture into increasingly dangerous wilderness. Exploration activities generate nascent
skills - swimming develops water affinity, falling repeatedly builds flight resistance, crafting
creates tool mastery. When death occurs, items vanish but skills mature into permanent abilities.""",

    """Echoes of the Fallen uses Voxel Plugin 2.0 for UE5 with Nanite and Lumen support. The architecture
uses Dual Contouring for mesh generation which preserves sharp edges for architectural elements.
Chunk size is 32x32x32 blocks with aggressive LOD, async chunk generation on background threads,
greedy meshing, frustum and occlusion culling, and vertex pooling across chunks for up to 40% performance gain.""",

    """Echoes of the Fallen skill system has action-based learning: swimming develops water magic affinity,
repeated falling builds gliding abilities, combat creates weapon masteries. Skills have two states:
Fresh skills appear during gameplay and become Fully Acquired upon death. Multiple skills create
hybrid abilities like water plus earth equals mud manipulation. The system is inspired by isekai
anime concepts of reincarnation with retained knowledge.""",

    """Echoes of the Fallen item system uses a temporary item philosophy: dungeon loot provides significant
immediate advantages but vanishes on death. Player-crafted items are weaker but more persistent,
becoming damaged rather than destroyed. Broken chest items can be dismantled for valuable crafting
components. This creates risk-reward decisions about using powerful items now versus saving them.""",

    """Echoes of the Fallen world generation uses Mixed Persistence with Handcrafted Core. The home meadow
has a 1km radius that is completely static with player building area. The exploration ring extends
to 5km with semi-procedural landmark preservation. The wilderness is unlimited and fully procedural
with increasing difficulty. Spatial difficulty scaling is inspired by Valheim with distance-based
challenge rather than level-based progression.""",

    """Echoes of the Fallen art direction follows a Beautiful Decay aesthetic - nature reclaiming civilization.
Color palette uses desaturated earth tones with warm accent highlights. Assets are created with
MagicaVoxel for rapid prototyping, Blender for complex animations, and optionally Qubicle for
advanced export. Audio uses nature soundscapes with procedural ambient tracks that respond to
player actions and environment.""",

    """Echoes of the Fallen development timeline is 18 months solo. Phase 1 Foundation covers months 1-6
including C++ learning, Voxel Plugin prototyping, and core systems. Phase 2 Content Development
covers months 7-12 with procedural biome generation, 3-5 distinct biomes, NPC systems, and
crafting. Phase 3 Polish and Launch covers months 13-18 with alpha testing, beta, marketing,
and platform submission. Total budget is approximately $1000-2000.""",

    """Echoes of the Fallen launch targets include 7000+ Steam wishlists before early access,
80+ Metacritic score, and $50,000+ first-year revenue. Long-term goals are 100,000+ lifetime
sales and 500+ active Discord community members. The game's unique value proposition is the only
game combining voxel exploration with persistent skill-based progression and isekai-inspired death
mechanics. Direct competitor is Bioprototype, indirect competitors include Dead Cells, Hades,
Subnautica, and Valheim.""",

    """Echoes of the Fallen multiplayer is planned for future development after single-player proves
viable. Architecture uses server-authoritative model with UE5 built-in replication, chunk-based
replication with Fast Array Replication, distance-based relevance limiting, and client-side
prediction. Strategy is to build single-player first, add 2-4 player co-op later, use AI bots
to reduce community dependency risks.""",
]

for i, section in enumerate(sections):
    print(f"\n--- Section {i+1}/{len(sections)} ---")
    add(section)
    time.sleep(1)  # Small delay to avoid overwhelming Ollama

print("\n" + "="*60)
print("Waiting 20s for async graph extraction...")
print("="*60)
time.sleep(20)

# Final audit
print("\n" + "="*60)
print("Final database audit")
print("="*60)

vc = requests.get(f"http://localhost:6333/collections/openmemory").json()["result"]["points_count"]
gs = requests.post("http://localhost:8474/db/neo4j/tx/commit",
    json={"statements": [
        {"statement": "MATCH (n) RETURN count(n) as entities"},
    ]}, auth=("neo4j", "openmemory")).json()
entities = gs["results"][0]["data"][0]["row"][0]

rels = requests.post("http://localhost:8474/db/neo4j/tx/commit",
    json={"statements": [
        {"statement": "MATCH ()-[r]->() RETURN count(r) as rels"},
    ]}, auth=("neo4j", "openmemory")).json()["results"][0]["data"][0]["row"][0]

self_refs = requests.post("http://localhost:8474/db/neo4j/tx/commit",
    json={"statements": [
        {"statement": "MATCH (n)-[r]->(n) RETURN count(r) as c"},
    ]}, auth=("neo4j", "openmemory")).json()["results"][0]["data"][0]["row"][0]

print(f"Qdrant vectors: {vc}")
print(f"Neo4j entities: {entities}")
print(f"Neo4j relationships: {rels}")
print(f"Self-referential edges: {self_refs}")

# Test a few searches
print("\n--- Search verification ---")
for query in [
    "what GPU does steven use",
    "echoes of the fallen skill system",
    "steven music production",
    "steven cat",
    "voxel chunk optimization",
]:
    r = requests.post(f"{API}/api/v1/memories/search",
        json={"query": query, "user_id": USER, "limit": 3}).json()
    results = r.get("results", [])
    sources = r.get("sources_used", [])
    top = results[0]["memory"][:60] if results else "NO RESULTS"
    score = results[0]["score"] if results else 0
    print(f"  '{query}': {len(results)} results ({'+'.join(sources)}), top={score:.2f}: {top}...")

print("\nDone!")
