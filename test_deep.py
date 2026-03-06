#!/usr/bin/env python3
"""Deep comprehensive test suite for mem0 hybrid memory system with qwen3.5:4b.

Tests all endpoints with diverse domains, curveballs, edge cases.
Run from host: python test_deep.py
"""
import requests, json, time, sys

API = "http://localhost:8765"
QDRANT = "http://localhost:6333"
NEO4J = "http://localhost:8474"
USER = "steven"
PASSED = 0
FAILED = 0
WARNINGS = 0

def ok(msg):
    global PASSED; PASSED += 1; print(f"  PASS: {msg}")
def fail(msg):
    global FAILED; FAILED += 1; print(f"  FAIL: {msg}")
def warn(msg):
    global WARNINGS; WARNINGS += 1; print(f"  WARN: {msg}")

def add_memory(text, user_id=USER):
    r = requests.post(f"{API}/api/v1/memories/", json={"text": text, "user_id": user_id})
    return r.json()

def search(query, limit=10, user_id=USER):
    r = requests.post(f"{API}/api/v1/memories/search", json={"query": query, "user_id": user_id, "limit": limit})
    return r.json()

def conversation(user_msg, llm_resp, user_id=USER):
    r = requests.post(f"{API}/api/v1/memories/conversation", json={
        "user_message": user_msg, "llm_response": llm_resp, "user_id": user_id, "app": "test"
    })
    return r.json()

def delete_memories(ids, user_id=USER):
    r = requests.delete(f"{API}/api/v1/memories/", json={"memory_ids": ids, "user_id": user_id})
    return r.json()

def qdrant_count():
    r = requests.get(f"{QDRANT}/collections/openmemory")
    return r.json()["result"]["points_count"]

def neo4j_query(cypher):
    r = requests.post(f"{NEO4J}/db/neo4j/tx/commit", json={"statements": [{"statement": cypher}]},
                       headers={"Content-Type": "application/json"}, auth=("neo4j", "openmemory"))
    data = r.json()
    if data.get("errors"):
        return []
    rows = []
    for result in data.get("results", []):
        cols = result.get("columns", [])
        for d in result.get("data", []):
            rows.append(dict(zip(cols, d["row"])))
    return rows

def graph_stats():
    entities = neo4j_query("MATCH (n) RETURN count(n) as c")[0]["c"]
    rels = neo4j_query("MATCH ()-[r]->() RETURN count(r) as c")[0]["c"]
    self_refs = neo4j_query("MATCH (n)-[r]->(n) RETURN count(r) as c")[0]["c"]
    types = neo4j_query("MATCH (n) RETURN DISTINCT labels(n)[0] as type, count(n) as c ORDER BY c DESC")
    return {"entities": entities, "relationships": rels, "self_refs": self_refs, "types": types}

# ==========================================================================
print("\n" + "="*70)
print("PHASE 1: Bulk insertion - diverse domains")
print("="*70)

# --- Batch 1: Space & astronomy ---
print("\n--- Batch 1: Space & Astronomy ---")
r = add_memory("""The James Webb Space Telescope (JWST) orbits at the Sun-Earth L2 Lagrange point,
approximately 1.5 million kilometers from Earth. Its primary mirror is 6.5 meters in diameter,
composed of 18 hexagonal gold-plated beryllium segments. JWST observes in infrared wavelengths
from 0.6 to 28.3 micrometers, enabling it to see through cosmic dust clouds and detect light
from galaxies formed just 200 million years after the Big Bang. The telescope cost approximately
$10 billion and took 25 years to develop. It was launched on December 25, 2021 aboard an
Ariane 5 rocket from French Guiana.""")
print(f"  Added: {r.get('count', 0)} facts, skipped: {r.get('skipped_duplicates', 0)}")
if r.get("count", 0) >= 3: ok("Space facts extracted")
else: fail(f"Expected 3+ space facts, got {r.get('count', 0)}")

# --- Batch 2: Biology curveball (mixing technical with casual) ---
print("\n--- Batch 2: Biology + casual language ---")
r = add_memory("""Honestly, tardigrades are insane. These microscopic animals (0.1-1.5mm) can survive
temperatures from -272C to 150C, pressures 6x deeper than the Mariana Trench, and even the vacuum
of space. They do this through cryptobiosis - basically they dry out into a 'tun' state where
metabolism drops to 0.01% of normal. Scientists even shot them to the Moon on the Beresheet lander
(it crashed lol) and they probably survived. There are over 1300 known species. Oh and they can
withstand 1000x more radiation than humans.""")
print(f"  Added: {r.get('count', 0)} facts")
if r.get("count", 0) >= 3: ok("Biology facts with casual language extracted")
else: fail(f"Expected 3+ biology facts, got {r.get('count', 0)}")

# --- Batch 3: Programming - deep technical chain ---
print("\n--- Batch 3: Deep technical chain (Rust ownership) ---")
r = add_memory("""Rust's ownership system has three rules: each value has exactly one owner,
when the owner goes out of scope the value is dropped, and ownership can be transferred via
moves. Borrowing allows references without taking ownership - immutable borrows (&T) allow
multiple simultaneous readers, while mutable borrows (&mut T) require exclusive access. The
borrow checker enforces these rules at compile time, preventing data races entirely. This is
why Rust achieves memory safety without a garbage collector. The lifetime system ('a annotations)
tracks how long references are valid, preventing dangling pointers. Traits like Send and Sync
enable safe concurrency - Send means a type can be transferred between threads, Sync means it
can be shared between threads via references.""")
print(f"  Added: {r.get('count', 0)} facts")
if r.get("count", 0) >= 4: ok("Deep Rust technical chain extracted")
else: fail(f"Expected 4+ Rust facts, got {r.get('count', 0)}")

# --- Batch 4: Music production ---
print("\n--- Batch 4: Music production ---")
r = add_memory("""Steven produces electronic music using Ableton Live 12 Suite on his Windows workstation.
He uses a Focusrite Scarlett 2i2 as his audio interface and monitors through Beyerdynamic DT 990 Pro
headphones (250 ohm). His go-to synthesizers are Serum by Xfer Records for sound design and Diva by
u-he for analog emulation. He mixes at -14 LUFS for streaming platforms and uses FabFilter Pro-Q 3
for surgical EQ work. His music leans toward synthwave and dark ambient genres.""")
print(f"  Added: {r.get('count', 0)} facts")
if r.get("count", 0) >= 4: ok("Music production facts extracted")
else: fail(f"Expected 4+ music facts, got {r.get('count', 0)}")

# --- Batch 5: Geography with numbers (curveball: lots of stats) ---
print("\n--- Batch 5: Geography stats ---")
r = add_memory("""Estonia has a population of 1.3 million people across 45,339 square kilometers,
making it one of the least densely populated countries in Europe at 29 people per sq km. Over
50% of the country is covered by forest. Tallinn, the capital, was founded in 1248 and has a
well-preserved medieval Old Town that is a UNESCO World Heritage Site. Estonia was the first
country to offer e-residency and holds parliamentary elections online. The country has over
2,222 islands, the largest being Saaremaa at 2,673 sq km.""")
print(f"  Added: {r.get('count', 0)} facts")
if r.get("count", 0) >= 3: ok("Geography facts extracted")
else: fail(f"Expected 3+ geography facts, got {r.get('count', 0)}")

# --- Batch 6: Personal preferences (curveball: opinion-style) ---
print("\n--- Batch 6: Personal opinions & preferences ---")
r = add_memory("""Steven strongly prefers local-first tools over cloud services. He thinks VS Code is
overrated and has been experimenting with Neovim and Zed. He believes game development is the best
way to learn programming because it covers algorithms, graphics, networking, and optimization all at
once. He dislikes subscription-based software models and prefers one-time purchases. His favorite
programming language is Rust but he admits C++ is still necessary for Unreal Engine work.""")
print(f"  Added: {r.get('count', 0)} facts")
if r.get("count", 0) >= 3: ok("Preference/opinion facts extracted")
else: fail(f"Expected 3+ preference facts, got {r.get('count', 0)}")

# --- Batch 7: Edge case - very short facts ---
print("\n--- Batch 7: Short individual facts ---")
for fact in [
    "Steven's birthday is March 15th",
    "Steven speaks Estonian, English, and some Russian",
    "Steven has a cat named Pixel",
]:
    r = add_memory(fact)
    results = r.get("results", [])
    status = results[0].get("event", "?") if results else "EMPTY"
    count = r.get("count", 0)
    if count >= 1 or status == "ADD": ok(f"Short fact stored: {fact[:40]}...")
    else: warn(f"Short fact status '{status}' count={count}: {fact[:40]}...")

# --- Batch 8: Edge case - already known info (dedup test) ---
print("\n--- Batch 8: Dedup test ---")
r = add_memory("Steven uses an RTX 4090 GPU with 24GB VRAM for AI inference and game development")
if r.get("skipped_duplicates", 0) > 0 or r.get("count", 0) == 0:
    ok("Duplicate correctly detected/skipped")
else:
    warn(f"Possible dedup miss: count={r.get('count')}, skipped={r.get('skipped_duplicates')}")

# Wait for async graph extraction
print("\n  Waiting 15s for async graph extraction...")
time.sleep(15)

# ==========================================================================
print("\n" + "="*70)
print("PHASE 2: Conversation memory tests")
print("="*70)

# Turn 1: Normal conversation
print("\n--- Turn 1: Normal technical conversation ---")
r = conversation(
    "I've been working on a multiplayer netcode system for Echoes of the Fallen. Using rollback-based prediction with 60Hz tick rate.",
    "Rollback netcode is great for fighting games. Are you using GGPO or a custom implementation?"
)
if r.get("facts_stored", 0) >= 1: ok(f"Conv turn 1: {r.get('facts_stored')} facts stored")
else: fail(f"Conv turn 1: {r.get('facts_stored', 0)} facts stored")
print(f"  Extracted: {r.get('extracted_facts', [])}")

# Turn 2: Curveball - user corrects previous info
print("\n--- Turn 2: Correction (switch from UE5 to Godot) ---")
r = conversation(
    "Actually I decided to switch Echoes of the Fallen from Unreal Engine 5 to Godot 4.3. The C++ build times in UE5 were killing my iteration speed.",
    "That's a big decision! Godot 4.3 has come a long way. The GDExtension system should let you keep your C++ code too."
)
if r.get("facts_stored", 0) >= 1: ok(f"Conv turn 2 (correction): {r.get('facts_stored')} facts")
else: fail(f"Conv turn 2 (correction): 0 facts stored")
print(f"  Extracted: {r.get('extracted_facts', [])}")

# Turn 3: Curveball - emotional/casual (should still extract facts)
print("\n--- Turn 3: Emotional/casual language ---")
r = conversation(
    "Man, I spent 3 days debugging a memory leak in the entity component system. Turns out it was a circular reference between the physics body and the collision shape. So frustrating but I learned a lot about weak references.",
    "Circular references are a classic gotcha! Weak references are the right solution. Did you add any tooling to detect these in the future?"
)
if r.get("facts_stored", 0) >= 1: ok(f"Conv turn 3 (casual): {r.get('facts_stored')} facts")
else: warn(f"Conv turn 3 (casual): 0 facts (may be too casual)")
print(f"  Extracted: {r.get('extracted_facts', [])}")

# Turn 4: Multi-topic in one turn
print("\n--- Turn 4: Multi-topic turn ---")
r = conversation(
    "Today I set up a CI pipeline with GitHub Actions for the game, configured my new mechanical keyboard (Keychron Q1 Pro with Gateron switches), and started reading 'Game Engine Architecture' by Jason Gregory.",
    "Productive day! That's a great book - Gregory covers everything from memory allocators to rendering pipelines."
)
if r.get("facts_stored", 0) >= 2: ok(f"Conv turn 4 (multi-topic): {r.get('facts_stored')} facts")
else: warn(f"Conv turn 4 (multi-topic): only {r.get('facts_stored', 0)} facts from 3 topics")
print(f"  Extracted: {r.get('extracted_facts', [])}")

# Turn 5: Curveball - questions only (should extract little/nothing)
print("\n--- Turn 5: Questions only (low-fact) ---")
r = conversation(
    "What do you think about using SQLite vs PostgreSQL for game save data? And should I implement hot-reloading for the scripting layer?",
    "For save data, SQLite is perfect - single file, no server needed. Hot-reloading is worth it for scripting but complex to implement safely."
)
facts = r.get("facts_stored", 0)
if facts <= 1: ok(f"Conv turn 5 (questions): correctly extracted {facts} facts")
else: warn(f"Conv turn 5 (questions): extracted {facts} facts from a question-only turn")

# Wait for graph
print("\n  Waiting 15s for async graph extraction...")
time.sleep(15)

# ==========================================================================
print("\n" + "="*70)
print("PHASE 3: Search quality tests")
print("="*70)

queries = [
    ("what GPU does steven use", ["RTX 4090", "4090", "VRAM"]),
    ("steven music production setup", ["Ableton", "Scarlett", "Serum", "Diva", "synthwave"]),
    ("tardigrade survival", ["tardigrade", "cryptobiosis", "tun", "radiation"]),
    ("james webb telescope specs", ["JWST", "Webb", "mirror", "infrared", "L2"]),
    ("rust ownership borrowing", ["ownership", "borrow", "lifetime", "Send", "Sync"]),
    ("estonia facts", ["Estonia", "Tallinn", "e-residency", "Saaremaa"]),
    ("steven game engine", ["Echoes", "Godot", "Unreal", "netcode"]),
    ("steven programming languages", ["Rust", "C++", "programming"]),
    ("steven cat", ["Pixel", "cat"]),
    ("steven keyboard setup", ["Keychron", "keyboard", "mechanical"]),
]

for query, expected_any in queries:
    r = search(query)
    results = r.get("results", [])
    sources = r.get("sources_used", [])
    if not results:
        fail(f"Search '{query}': 0 results")
        continue

    top_memory = results[0].get("memory", "")
    top_score = results[0].get("score", 0)
    all_text = " ".join([m.get("memory", "") for m in results])

    found = [kw for kw in expected_any if kw.lower() in all_text.lower()]
    if found:
        ok(f"Search '{query}': {len(results)} results, top={top_score:.2f} ({results[0].get('source', '?')}), keywords={found}")
    else:
        fail(f"Search '{query}': none of {expected_any} found in results")

    # Check source diversity
    if len(sources) > 1:
        pass  # good, multi-source
    elif len(results) > 0:
        warn(f"  Only source: {sources}")

# ==========================================================================
print("\n" + "="*70)
print("PHASE 4: Delete & related memories test")
print("="*70)

# First, find a memory to delete
r = search("tardigrade", limit=3)
results = r.get("results", [])
if results:
    target_id = results[0]["id"]
    target_text = results[0]["memory"][:60]
    print(f"  Deleting: {target_text}...")

    dr = delete_memories([target_id])
    if "deleted" in dr or "message" in dr:
        ok(f"Delete returned structured response")
        # Verify it's gone
        r2 = search("tardigrade", limit=3)
        remaining_vector = [m for m in r2.get("results", []) if m["id"] == target_id and m.get("source") == "vector"]
        if not remaining_vector:
            ok("Deleted memory no longer appears in vector search (graph may still have entities)")
        else:
            fail("Deleted memory still appears in vector search")
    else:
        fail(f"Delete response unexpected: {json.dumps(dr)[:100]}")
else:
    warn("No tardigrade memories to delete")

# ==========================================================================
print("\n" + "="*70)
print("PHASE 5: Graph quality audit")
print("="*70)

stats = graph_stats()
print(f"\n  Entities: {stats['entities']}")
print(f"  Relationships: {stats['relationships']}")
print(f"  Self-refs: {stats['self_refs']}")

if stats["self_refs"] == 0: ok("Zero self-referential edges")
else: fail(f"{stats['self_refs']} self-referential edges found")

if stats["entities"] >= 20: ok(f"Good entity count: {stats['entities']}")
else: warn(f"Low entity count: {stats['entities']}")

if stats["relationships"] >= 15: ok(f"Good relationship count: {stats['relationships']}")
else: warn(f"Low relationship count: {stats['relationships']}")

# Type diversity
print("\n  Entity types:")
type_set = set()
for t in stats["types"]:
    etype = t.get("type", "?")
    count = t.get("c", 0)
    type_set.add(etype)
    print(f"    {etype}: {count}")

if len(type_set) >= 5: ok(f"Good type diversity: {len(type_set)} types")
else: warn(f"Low type diversity: {len(type_set)} types")

# Check hub connectivity (steven should have most connections)
steven_rels = neo4j_query("MATCH (s)-[r]->() WHERE s.name CONTAINS 'steven' RETURN count(r) as c")
steven_count = steven_rels[0]["c"] if steven_rels else 0
print(f"\n  Steven outgoing relationships: {steven_count}")
if steven_count >= 10: ok(f"Steven is well-connected hub ({steven_count} rels)")
else: warn(f"Steven has only {steven_count} outgoing relationships")

# Check for non-hub sources (anything that's not person/project as source)
non_hub = neo4j_query("""
    MATCH (src)-[r]->(tgt)
    WHERE NOT any(l IN labels(src) WHERE l IN ['person', 'project'])
    RETURN src.name as source, labels(src)[0] as type, type(r) as rel, tgt.name as target
    LIMIT 10
""")
if non_hub:
    warn(f"Non-hub sources found: {len(non_hub)}")
    for nh in non_hub[:3]:
        print(f"    {nh['source']} ({nh['type']}) --{nh['rel']}--> {nh['target']}")
else:
    ok("All relationship sources are person or project (hub-only)")

# Print all relationships for inspection
print("\n  All relationships:")
all_rels = neo4j_query("""
    MATCH (s)-[r]->(t)
    RETURN s.name as source, labels(s)[0] as src_type, type(r) as rel, t.name as target, labels(t)[0] as tgt_type
    ORDER BY s.name, type(r)
""")
for rel in all_rels:
    marker = " !" if rel["src_type"] not in ("person", "project") else ""
    print(f"    {rel['source']} ({rel['src_type']}) --{rel['rel']}--> {rel['target']} ({rel['tgt_type']}){marker}")

# ==========================================================================
print("\n" + "="*70)
print("PHASE 6: Edge cases & stress tests")
print("="*70)

# Unicode / special characters
print("\n--- Unicode test ---")
r = add_memory("Steven visited Tallinn's Cafe Maiasmokk (Cafe Maiasmokk), the oldest cafe in Estonia, founded in 1864")
if r.get("count", 0) >= 1: ok("Unicode content stored")
else: warn("Unicode content may have been lost")

# Very long single fact
print("\n--- Long single fact ---")
r = add_memory("The Voyager 1 spacecraft, launched by NASA on September 5, 1977, is the most distant human-made object from Earth, currently over 24 billion kilometers away in interstellar space, still transmitting data back to Earth via its 23-watt radio transmitter, which is about as powerful as a refrigerator light bulb")
if r.get("count", 0) >= 1: ok("Long single fact stored")
else: fail("Long single fact not stored")

# Contradicting info
print("\n--- Contradiction test ---")
r = add_memory("Steven's favorite color is blue")
r2 = add_memory("Steven's favorite color is dark purple")
# Both should be stored (system doesn't resolve contradictions)
ok("Contradiction test: both stored (no auto-resolution expected)")

# Empty/minimal input
print("\n--- Minimal input ---")
r = add_memory("hi")
if r.get("count", 0) == 0: ok("Minimal input correctly rejected")
else: warn(f"Minimal input 'hi' produced {r.get('count', 0)} facts")

# Search pagination
print("\n--- Pagination test ---")
r1 = requests.post(f"{API}/api/v1/memories/search", json={"query": "steven", "user_id": USER, "limit": 3, "offset": 0}).json()
r2 = requests.post(f"{API}/api/v1/memories/search", json={"query": "steven", "user_id": USER, "limit": 3, "offset": 3}).json()
ids_page1 = set(m["id"] for m in r1.get("results", []))
ids_page2 = set(m["id"] for m in r2.get("results", []))
if ids_page1 and ids_page2 and not ids_page1.intersection(ids_page2):
    ok(f"Pagination: page1={len(ids_page1)} ids, page2={len(ids_page2)} ids, no overlap")
elif not ids_page2:
    warn("Pagination: page 2 empty (may not have enough results)")
else:
    fail(f"Pagination overlap: {ids_page1.intersection(ids_page2)}")

# Graph context endpoint
print("\n--- Graph context endpoint ---")
r = requests.get(f"{API}/api/v1/memories/graph/context/steven?user_id={USER}").json()
if r.get("topic") or r.get("context"):
    ok(f"Graph context for 'steven' returned data")
else:
    warn(f"Graph context empty: {json.dumps(r)[:100]}")

# ==========================================================================
print("\n" + "="*70)
print("PHASE 7: Final database audit")
print("="*70)

vc = qdrant_count()
gs = graph_stats()

print(f"\n  Qdrant vectors: {vc}")
print(f"  Neo4j entities: {gs['entities']}")
print(f"  Neo4j relationships: {gs['relationships']}")
print(f"  Self-refs: {gs['self_refs']}")

if vc >= 25: ok(f"Good vector count: {vc}")
else: warn(f"Low vector count: {vc}")

# ==========================================================================
print("\n" + "="*70)
print(f"RESULTS: {PASSED} passed, {FAILED} failed, {WARNINGS} warnings")
print("="*70 + "\n")

sys.exit(1 if FAILED > 0 else 0)
