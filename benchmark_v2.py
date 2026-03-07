#!/usr/bin/env python3
"""
Comprehensive v1 vs v2 benchmark.
Uses separate user IDs per phase to avoid slow delete_all calls.

Usage:
  python benchmark_v2.py [--api-url http://localhost:8765]
"""

import json
import sys
import time
import requests

# Fix Unicode on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API = "http://localhost:8765"
USER_V1 = "bench_v1"
USER_V2 = "bench_v2"

for i, arg in enumerate(sys.argv):
    if arg == "--api-url" and i + 1 < len(sys.argv):
        API = sys.argv[i + 1]

# ============================================================================
# Helpers
# ============================================================================

def v1_add(text, user_id=USER_V1):
    start = time.time()
    r = requests.post(f"{API}/api/v1/memories/", json={"text": text, "user_id": user_id}, timeout=120)
    elapsed = time.time() - start
    return r.json(), elapsed

def v1_search(query, user_id=USER_V1, limit=5):
    start = time.time()
    r = requests.post(f"{API}/api/v1/memories/search",
        json={"query": query, "user_id": user_id, "limit": limit}, timeout=30)
    elapsed = time.time() - start
    return r.json(), elapsed

def v2_brain(request, user_id=USER_V2):
    start = time.time()
    r = requests.post(f"{API}/api/v1/brain",
        json={"request": request, "user_id": user_id}, timeout=300)
    elapsed = time.time() - start
    return r.json(), elapsed

def ensure_user(user_id):
    """Create user by storing+deleting a dummy fact via brain agent."""
    try:
        r = requests.post(f"{API}/api/v1/brain",
            json={"request": f"Remember: {user_id} is a test user", "user_id": user_id},
            timeout=120)
        return r.status_code == 200
    except:
        return False

def db_stats():
    stats = {}
    try:
        qdrant = requests.get("http://localhost:6333/collections/openmemory", timeout=5).json()
        stats["vectors"] = qdrant["result"]["points_count"]
    except:
        stats["vectors"] = "?"

    try:
        neo = requests.post("http://localhost:8474/db/neo4j/tx/commit",
            json={"statements": [{"statement": "MATCH (n) RETURN count(n) as c"}]},
            auth=("neo4j", "openmemory"), timeout=5).json()
        stats["entities"] = neo["results"][0]["data"][0]["row"][0]
    except:
        stats["entities"] = "?"

    try:
        neo2 = requests.post("http://localhost:8474/db/neo4j/tx/commit",
            json={"statements": [{"statement": "MATCH ()-[r]->() RETURN count(r) as c"}]},
            auth=("neo4j", "openmemory"), timeout=5).json()
        stats["relationships"] = neo2["results"][0]["data"][0]["row"][0]
    except:
        stats["relationships"] = "?"

    return stats

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def count_user_vectors(user_id):
    """Count Qdrant vectors for a specific user."""
    try:
        r = requests.post("http://localhost:6333/collections/openmemory/points/scroll",
            json={"filter": {"must": [{"key": "user_id", "match": {"value": user_id}}]},
                  "limit": 1000, "with_payload": False},
            timeout=10)
        return len(r.json().get("result", {}).get("points", []))
    except:
        return "?"


# ============================================================================
# Test Data
# ============================================================================

PERSONAL_FACTS = [
    "Steven lives in Tallinn, Estonia and works from home as a developer",
    "Steven's birthday is March 15th",
    "Steven has a cat named Pixel",
    "Steven speaks Estonian, English, and some Russian",
    "Steven strongly prefers local-first, open-source tools over cloud services",
    "Steven prefers dark mode in all applications",
    "Steven's favorite programming language is Rust but he also uses C++ for Unreal Engine work",
    "Steven has a background in PHP web development and is transitioning to game development",
    "Steven has an NVIDIA RTX 4090 GPU with 24GB VRAM in his Windows 11 workstation",
    "Steven produces electronic music using Ableton Live 12 Suite",
    "Steven's music production leans toward synthwave and dark ambient genres",
    "Steven uses a Focusrite Scarlett 2i2 audio interface",
    "Steven monitors audio through Beyerdynamic DT 990 Pro headphones at 250 ohms",
]

ECHOES_FACTS = [
    "Echoes of the Fallen is a voxel-based roguelike exploration game developed by Steven as a solo indie project",
    "Echoes of the Fallen uses Voxel Plugin 2.0 for UE5 with Nanite and Lumen support and Dual Contouring for mesh generation",
    "Echoes of the Fallen skill system: actions generate persistent skills that ripen after death. Swimming develops water affinity, falling builds flight resistance, crafting creates tool mastery",
    "Echoes of the Fallen item system: dungeon loot vanishes on death, player-crafted items become damaged not destroyed, broken chest items can be dismantled for crafting components",
    "Echoes of the Fallen world has a static home meadow (1km radius), semi-procedural exploration ring (5km), and unlimited procedural wilderness with distance-based difficulty like Valheim",
    "Echoes of the Fallen development timeline is 18 months solo with budget of $1000-2000. Phase 1: Foundation (months 1-6), Phase 2: Content (months 7-12), Phase 3: Polish and Launch (months 13-18)",
    "Echoes of the Fallen launch targets: 7000+ Steam wishlists, 80+ Metacritic, $50,000+ first-year revenue, 100,000+ lifetime sales",
    "Echoes of the Fallen chunk management: 32x32x32 blocks, aggressive LOD, async generation, greedy meshing, vertex pooling for 40% performance gain",
]

SEARCH_QUERIES = [
    ("What GPU does Steven use?", ["RTX 4090", "4090"]),
    ("What is Steven's cat's name?", ["Pixel"]),
    ("What game is Steven developing?", ["Echoes of the Fallen", "Echoes"]),
    ("What music does Steven produce?", ["synthwave", "dark ambient", "Ableton"]),
    ("How does the skill system work in Echoes of the Fallen?", ["ripen", "death", "swimming", "skill"]),
    ("What is the chunk size in Echoes of the Fallen?", ["32x32x32", "32"]),
    ("Where does Steven live?", ["Tallinn", "Estonia"]),
    ("What are the launch targets for Echoes of the Fallen?", ["7000", "wishlist", "Steam"]),
    ("What programming languages does Steven know?", ["Rust", "C++", "PHP"]),
    ("What headphones does Steven use?", ["DT 990", "Beyerdynamic", "250"]),
]


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark():
    print_header("MEMORY SYSTEM v1 vs v2 BENCHMARK")
    print(f"API: {API}")
    print(f"v1 user: {USER_V1}, v2 user: {USER_V2}")

    # Check API
    try:
        r = requests.get(f"{API}/docs", timeout=5)
        print(f"API status: {r.status_code}")
    except Exception as e:
        print(f"API not reachable: {e}")
        sys.exit(1)

    # Brain status
    try:
        status = requests.get(f"{API}/api/v1/brain/status", timeout=5).json()
        print(f"Brain model: {status.get('model')}, max_steps: {status.get('max_steps')}")
    except:
        print("Brain endpoint not available (v2 not deployed?)")

    before = db_stats()
    print(f"DB before: vectors={before['vectors']}, entities={before['entities']}, rels={before['relationships']}")

    # Create users via brain (auto-creates in SQLite)
    print("\nCreating test users...")
    for uid in [USER_V1, USER_V2]:
        ok = ensure_user(uid)
        print(f"  {uid}: {'ok' if ok else 'FAILED'}")

    time.sleep(2)

    # ==================================================================
    # PHASE 1: v1 INSERTION
    # ==================================================================
    print_header("PHASE 1: v1 INSERTION (direct REST API)")

    v1_times = []
    v1_added = 0

    all_facts = PERSONAL_FACTS + ECHOES_FACTS
    for i, fact in enumerate(all_facts):
        data, elapsed = v1_add(fact, user_id=USER_V1)
        v1_times.append(elapsed)
        count = data.get("count", 0)
        v1_added += count
        skipped = data.get("skipped_duplicates", 0)
        status_icon = "+" if count > 0 else "="
        print(f"  [{elapsed:.2f}s] {status_icon}{count} (skip {skipped}): {fact[:65]}...")

    print(f"\n  v1 Total: {v1_added} facts added in {len(v1_times)} calls")
    print(f"  v1 Time: {sum(v1_times):.1f}s total, {sum(v1_times)/max(len(v1_times),1):.2f}s avg")
    print(f"  v1 Range: {min(v1_times):.2f}s - {max(v1_times):.2f}s")

    # Wait for async graph extraction
    print("\n  Waiting 20s for async graph extraction...")
    time.sleep(20)

    v1_vectors = count_user_vectors(USER_V1)
    after_v1 = db_stats()
    print(f"  v1 user vectors: {v1_vectors}")
    print(f"  DB after v1: vectors={after_v1['vectors']}, entities={after_v1['entities']}, rels={after_v1['relationships']}")

    # ==================================================================
    # PHASE 2: v1 SEARCH
    # ==================================================================
    print_header("PHASE 2: v1 SEARCH (direct REST API)")

    v1_search_times = []
    v1_search_quality = 0

    for query, expected_terms in SEARCH_QUERIES:
        data, elapsed = v1_search(query, user_id=USER_V1, limit=5)
        v1_search_times.append(elapsed)
        results = data.get("results", [])

        all_text = " ".join(r.get("memory", "") for r in results).lower()
        found = sum(1 for term in expected_terms if term.lower() in all_text)
        quality = found / len(expected_terms) if expected_terms else 0
        v1_search_quality += quality

        top = results[0]["memory"][:60] if results else "NO RESULTS"
        top_score = results[0]["score"] if results else 0
        print(f"  [{elapsed:.3f}s] Q: {query[:50]}...")
        print(f"    {len(results)} results, top={top_score:.3f}, quality={quality:.0%}: {top}...")

    avg_quality_v1 = v1_search_quality / len(SEARCH_QUERIES)
    print(f"\n  v1 Search: {sum(v1_search_times):.2f}s total, {sum(v1_search_times)/len(SEARCH_QUERIES):.3f}s avg")
    print(f"  v1 Quality: {avg_quality_v1:.0%} average term match")

    # ==================================================================
    # PHASE 3: v2 INSERTION
    # ==================================================================
    print_header("PHASE 3: v2 INSERTION (brain agent)")

    v2_insert_times = []
    v2_stored = 0

    # Insert facts one by one for fair comparison
    for i, fact in enumerate(all_facts):
        prompt = f"Remember this: {fact}"
        data, elapsed = v2_brain(prompt, user_id=USER_V2)
        v2_insert_times.append(elapsed)

        steps = data.get("steps", "?")
        tools = data.get("tools_called", [])
        success = data.get("success", False)
        stores = tools.count("vector_store") if isinstance(tools, list) else 0
        v2_stored += stores
        status_icon = "+" if stores > 0 else "x"

        print(f"  [{elapsed:.1f}s] {status_icon}{stores} steps={steps}: {fact[:60]}...")

    print(f"\n  v2 Total: {v2_stored} facts stored in {len(v2_insert_times)} calls")
    print(f"  v2 Time: {sum(v2_insert_times):.1f}s total, {sum(v2_insert_times)/max(len(v2_insert_times),1):.1f}s avg")
    print(f"  v2 Range: {min(v2_insert_times):.1f}s - {max(v2_insert_times):.1f}s")

    # Wait for async graph extraction
    print("\n  Waiting 20s for async graph extraction...")
    time.sleep(20)

    v2_vectors = count_user_vectors(USER_V2)
    after_v2 = db_stats()
    print(f"  v2 user vectors: {v2_vectors}")
    print(f"  DB after v2: vectors={after_v2['vectors']}, entities={after_v2['entities']}, rels={after_v2['relationships']}")

    # ==================================================================
    # PHASE 4: v2 SEARCH
    # ==================================================================
    print_header("PHASE 4: v2 SEARCH (brain agent)")

    v2_search_times = []
    v2_search_quality = 0

    for query, expected_terms in SEARCH_QUERIES:
        data, elapsed = v2_brain(query, user_id=USER_V2)
        v2_search_times.append(elapsed)

        answer = data.get("answer", "")
        steps = data.get("steps", "?")
        tools = data.get("tools_called", [])

        found = sum(1 for term in expected_terms if term.lower() in answer.lower())
        quality = found / len(expected_terms) if expected_terms else 0
        v2_search_quality += quality

        print(f"  [{elapsed:.1f}s] Q: {query[:50]}...")
        print(f"    steps={steps}, tools={tools}, quality={quality:.0%}")
        print(f"    Answer: {answer[:150]}")

    avg_quality_v2 = v2_search_quality / len(SEARCH_QUERIES)
    print(f"\n  v2 Search: {sum(v2_search_times):.1f}s total, {sum(v2_search_times)/len(SEARCH_QUERIES):.1f}s avg")
    print(f"  v2 Quality: {avg_quality_v2:.0%} average term match")

    # ==================================================================
    # PHASE 5: v2 COMPLEX OPERATIONS
    # ==================================================================
    print_header("PHASE 5: v2 COMPLEX OPERATIONS")

    complex_ops = [
        ("Cross-DB synthesis", "Tell me everything you know about Steven's hardware setup"),
        ("Relationship query", "How is Steven connected to Echoes of the Fallen?"),
        ("Update operation", "Steven upgraded his GPU from RTX 4090 to RTX 5090 with 32GB VRAM"),
        ("Targeted delete", "Delete all memories about music production"),
        ("Verify delete", "Does Steven produce any music?"),
        ("Multi-fact store", "Steven adopted a second cat named Luna. He also started learning Japanese."),
        ("Verify store", "What pets does Steven have?"),
    ]

    complex_times = []
    for label, request in complex_ops:
        data, elapsed = v2_brain(request, user_id=USER_V2)
        complex_times.append(elapsed)
        answer = data.get("answer", "")[:200]
        steps = data.get("steps", "?")
        tools = data.get("tools_called", [])
        print(f"\n  [{elapsed:.1f}s] {label}")
        print(f"    Steps: {steps}, Tools: {tools}")
        print(f"    Answer: {answer}")

    # ==================================================================
    # PHASE 6: DATABASE AUDIT
    # ==================================================================
    print_header("PHASE 6: DATABASE AUDIT")

    final = db_stats()
    v1_vecs = count_user_vectors(USER_V1)
    v2_vecs = count_user_vectors(USER_V2)
    print(f"  Qdrant total: {final['vectors']} (v1 user: {v1_vecs}, v2 user: {v2_vecs})")
    print(f"  Neo4j entities: {final['entities']}")
    print(f"  Neo4j relationships: {final['relationships']}")

    # Self-refs
    try:
        neo = requests.post("http://localhost:8474/db/neo4j/tx/commit",
            json={"statements": [{"statement": "MATCH (n)-[r]->(n) RETURN count(r) as c"}]},
            auth=("neo4j", "openmemory"), timeout=5).json()
        print(f"  Self-referential edges: {neo['results'][0]['data'][0]['row'][0]}")
    except:
        print("  Self-referential edges: ?")

    # Entity types
    try:
        neo = requests.post("http://localhost:8474/db/neo4j/tx/commit",
            json={"statements": [{"statement": "MATCH (n) RETURN labels(n)[0] as type, count(n) as c ORDER BY c DESC LIMIT 10"}]},
            auth=("neo4j", "openmemory"), timeout=5).json()
        print(f"\n  Entity types:")
        for row in neo["results"][0]["data"]:
            t, c = row["row"]
            print(f"    {t}: {c}")
    except:
        pass

    # Sample relationships
    try:
        neo = requests.post("http://localhost:8474/db/neo4j/tx/commit",
            json={"statements": [{"statement": "MATCH (n)-[r]->(m) WHERE n.name CONTAINS 'steven' OR m.name CONTAINS 'echoes' RETURN n.name, type(r), m.name LIMIT 15"}]},
            auth=("neo4j", "openmemory"), timeout=5).json()
        print(f"\n  Key relationships:")
        for row in neo["results"][0]["data"]:
            src, rel, tgt = row["row"]
            print(f"    {src} --{rel}--> {tgt}")
    except:
        pass

    # ==================================================================
    # PHASE 7: SUMMARY
    # ==================================================================
    print_header("SUMMARY: v1 vs v2")

    print(f"\n  {'Metric':<35} {'v1 (direct)':<20} {'v2 (brain agent)':<20}")
    print(f"  {'-'*75}")

    # Insertion
    v1_total_insert = sum(v1_times)
    v2_total_insert = sum(v2_insert_times)
    print(f"  {'Facts inserted':<35} {v1_added:<20} {v2_stored:<20}")
    print(f"  {'Insert time (total)':<35} {v1_total_insert:.1f}s{'':<14} {v2_total_insert:.1f}s")
    if v1_added > 0 and v2_stored > 0:
        print(f"  {'Insert time (per fact)':<35} {v1_total_insert/v1_added:.2f}s{'':<14} {v2_total_insert/v2_stored:.1f}s")
    print(f"  {'Vectors stored':<35} {v1_vecs:<20} {v2_vecs:<20}")

    # Search
    v1_total_search = sum(v1_search_times)
    v2_total_search = sum(v2_search_times)
    print(f"  {'Search time (total, 10 queries)':<35} {v1_total_search:.2f}s{'':<14} {v2_total_search:.1f}s")
    print(f"  {'Search time (avg)':<35} {v1_total_search/len(SEARCH_QUERIES):.3f}s{'':<14} {v2_total_search/len(SEARCH_QUERIES):.1f}s")
    print(f"  {'Search quality':<35} {avg_quality_v1:.0%}{'':<18} {avg_quality_v2:.0%}")

    # Speed ratio
    if v1_total_search > 0:
        print(f"\n  v2 is {v2_total_search/v1_total_search:.0f}x slower at search")
    if v1_total_insert > 0:
        print(f"  v2 is {v2_total_insert/v1_total_insert:.0f}x slower at insertion")

    # Complex ops
    if complex_times:
        print(f"\n  v2 Complex ops: {sum(complex_times):.1f}s total, {sum(complex_times)/len(complex_times):.1f}s avg")

    print(f"\n{'='*70}")
    print(f"  Benchmark complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_benchmark()
