"""
Comprehensive quality test for v2 (brain agent).
Tests: insert, search, dedup, update, delete+graph cleanup, graph quality, edge cases.
Inspects Qdrant and Neo4j directly for DB-level verification.
"""
import requests
import time
import json
import sys
from datetime import datetime

BASE = "http://localhost:8765"
USER = f"bigtest_{datetime.now().strftime('%H%M%S')}"
LOG_FILE = f"big_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

def brain(request, timeout=120):
    r = requests.post(f"{BASE}/api/v1/brain", json={"request": request, "user_id": USER}, timeout=timeout)
    return r.json()

def qdrant_count():
    """Count vectors in Qdrant for this user"""
    try:
        r = requests.post("http://localhost:6333/collections/openmemory/points/scroll", json={
            "filter": {"must": [{"key": "user_id", "match": {"value": USER}}]},
            "limit": 500, "with_payload": True
        }, timeout=10)
        points = r.json().get("result", {}).get("points", [])
        return len(points), [p["payload"].get("data", "")[:80] for p in points]
    except:
        return -1, []

def neo4j_query(cypher):
    """Direct Neo4j query"""
    try:
        r = requests.post("http://localhost:8474/db/neo4j/tx/commit", json={
            "statements": [{"statement": cypher, "parameters": {"uid": USER}}]
        }, headers={"Authorization": "Basic bmVvNGo6b3Blbm1lbW9yeQ=="}, timeout=10)
        data = r.json()
        results = []
        for res in data.get("results", []):
            for row in res.get("data", []):
                results.append(row.get("row", []))
        return results
    except Exception as e:
        return [f"error: {e}"]

def neo4j_nodes():
    return neo4j_query(f"MATCH (n) WHERE n.user_id = $uid RETURN n.name, labels(n), n.mentions ORDER BY n.name")

def neo4j_edges():
    return neo4j_query(f"MATCH (n)-[r]->(m) WHERE n.user_id = $uid RETURN n.name, type(r), m.name ORDER BY n.name, type(r)")

def neo4j_self_refs():
    return neo4j_query(f"MATCH (n)-[r]->(n) WHERE n.user_id = $uid RETURN n.name, type(r)")

# ============================================================
# Wait for API
# ============================================================
log("Waiting for API...")
for i in range(20):
    try:
        requests.get(f"{BASE}/docs", timeout=5)
        break
    except:
        time.sleep(2)

# Ensure user exists
brain("Hello")
time.sleep(1)

log(f"\n{'='*70}")
log(f"  BIG QUALITY TEST — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Branch: memory_agent (v2), User: {USER}")
log(f"{'='*70}\n")

# ============================================================
# PHASE 1: INSERT — diverse categories
# ============================================================
log("PHASE 1: INSERT (diverse data)")
log("-" * 50)

inserts = [
    # Personal
    ("Steven was born in Tallinn, Estonia", "personal"),
    ("Steven speaks Estonian and English fluently", "personal"),
    ("Steven has a girlfriend named Mirjam who is a graphic designer", "personal"),
    ("Steven's favorite food is sushi, especially salmon nigiri", "personal"),

    # Technical / Work
    ("Steven works as a full-stack developer specializing in TypeScript and Rust", "tech"),
    ("Steven's main workstation has an RTX 4090 GPU with 24GB VRAM", "hardware"),
    ("Steven runs Docker on Windows 11 with WSL2 for development", "tech"),
    ("Steven prefers Neovim as his primary code editor with lazy.nvim config", "tech"),
    ("Steven uses Ollama for running local LLMs and avoids cloud AI services", "tech"),

    # Project — Echoes of the Fallen
    ("Echoes of the Fallen is a voxel-based adventure game built in Unreal Engine 5", "project"),
    ("Echoes of the Fallen uses dual contouring for terrain generation", "project"),
    ("The development timeline for Echoes of the Fallen is 18 months solo", "project"),
    ("Echoes of the Fallen has 4 biomes: forest, desert, tundra, and volcanic", "project"),

    # Music
    ("Steven plays guitar and piano, mostly jazz and blues", "music"),
    ("Steven's favorite band is Radiohead, favorite album is OK Computer", "music"),

    # Relationships / People
    ("Steven's friend Alex is a DevOps engineer at Cloudflare", "people"),
    ("Steven's mentor is Professor Tamm from Tallinn University of Technology", "people"),

    # Miscellaneous
    ("Steven has a cat named Pixel who is a grey British Shorthair", "misc"),
    ("Steven practices intermittent fasting with a 16:8 schedule", "misc"),
    ("Steven's home lab runs a Proxmox cluster with 3 mini PCs", "misc"),

    # Edge cases — compound facts
    ("Steven switched from VS Code to Neovim in 2024 and never looked back", "edge"),
    ("Steven tried Godot but went back to Unreal Engine because of Nanite support", "edge"),

    # Edge cases — negation
    ("Steven does not use macOS, only Windows and Linux", "edge"),
    ("Steven avoids JavaScript frameworks like React, preferring vanilla JS or htmx", "edge"),

    # Edge cases — numbers/versions
    ("Steven's home network runs at 10 Gbps with Ubiquiti UniFi gear", "edge"),
    ("Steven uses PostgreSQL 16 for production databases", "edge"),
]

for i, (fact, cat) in enumerate(inserts):
    r = brain(f"Remember: {fact}")
    answer = r.get("answer", "")[:80]
    steps = r.get("steps", "?")
    tools = r.get("tools_called", [])
    status = "OK" if r.get("success") else "FAIL"
    log(f"  [{i+1:2d}/{len(inserts)}] [{cat:8s}] {status} ({steps} steps) {fact[:55]}")
    time.sleep(0.3)

GRAPH_WAIT = 300  # ~10-15s per fact × 26 facts = 260-390s
log(f"\nWaiting {GRAPH_WAIT}s for queued graph extraction to complete...")
for remaining in range(GRAPH_WAIT, 0, -30):
    log(f"  ... {remaining}s remaining")
    time.sleep(min(30, remaining))

# Check DB counts
qdrant_n, qdrant_facts = qdrant_count()
nodes = neo4j_nodes()
edges = neo4j_edges()
log(f"\nDB State after insert:")
log(f"  Qdrant vectors: {qdrant_n}")
log(f"  Neo4j nodes: {len(nodes)}")
log(f"  Neo4j edges: {len(edges)}")

# ============================================================
# PHASE 2: SEARCH QUALITY
# ============================================================
log(f"\n\nPHASE 2: SEARCH QUALITY")
log("-" * 50)

searches = [
    ("What GPU does Steven use?", ["rtx 4090", "24gb", "vram"]),
    ("Where was Steven born?", ["tallinn", "estonia"]),
    ("What languages does Steven speak?", ["estonian", "english"]),
    ("Tell me about Echoes of the Fallen", ["voxel", "unreal", "adventure"]),
    ("What biomes does Echoes have?", ["forest", "desert", "tundra", "volcanic"]),
    ("What editor does Steven use?", ["neovim", "lazy"]),
    ("Does Steven have any pets?", ["pixel", "cat", "british"]),
    ("What music does Steven like?", ["radiohead", "guitar", "piano", "jazz"]),
    ("Who is Mirjam?", ["girlfriend", "graphic designer"]),
    ("What programming languages does Steven know?", ["typescript", "rust"]),
    ("Does Steven use macOS?", ["not", "no", "windows", "linux"]),
    ("What database does Steven use?", ["postgresql", "16"]),
    ("What's Steven's diet like?", ["fasting", "16:8", "intermittent"]),
    ("Who is Professor Tamm?", ["mentor", "tallinn"]),
    ("What's Steven's home network setup?", ["10 gbps", "ubiquiti", "unifi"]),
    ("Why did Steven choose Unreal over Godot?", ["nanite", "went back", "tried"]),
    ("What terrain tech does Echoes use?", ["dual contouring"]),
    ("Who is Alex?", ["devops", "cloudflare"]),
]

search_pass = 0
search_fail = 0
for query, expected in searches:
    r = brain(query)
    answer = r.get("answer", "").lower()
    tools = r.get("tools_called", [])
    steps = r.get("steps", "?")

    found = [kw for kw in expected if kw.lower() in answer]
    missed = [kw for kw in expected if kw.lower() not in answer]
    score = len(found) / len(expected) * 100

    status = "PASS" if score >= 50 else "FAIL"
    if status == "PASS":
        search_pass += 1
    else:
        search_fail += 1

    log(f"  [{status}] ({steps} steps, {score:3.0f}%) Q: {query}")
    if missed:
        log(f"         Missing: {missed}")
    if status == "FAIL":
        log(f"         Answer: {answer[:120]}")

log(f"\nSearch results: {search_pass}/{len(searches)} passed ({search_pass/len(searches)*100:.0f}%)")

# ============================================================
# PHASE 3: DEDUP
# ============================================================
log(f"\n\nPHASE 3: DEDUP TEST")
log("-" * 50)

before_count = qdrant_count()[0]

dedup_tests = [
    "Steven has an RTX 4090 GPU",  # exact duplicate
    "Steven uses a 4090 graphics card with 24GB",  # semantic duplicate
    "Echoes of the Fallen is built with Unreal Engine 5",  # semantic duplicate
    "Steven was born in Tallinn",  # shorter version of existing
]

for fact in dedup_tests:
    r = brain(f"Remember: {fact}")
    answer = r.get("answer", "")
    is_dup = any(x in answer.lower() for x in ["duplicate", "already", "exists", "stored", "similar"])
    log(f"  {'DUP' if is_dup else 'NEW'}: {fact[:60]} -> {answer[:80]}")

time.sleep(2)
after_count = qdrant_count()[0]
new_vectors = after_count - before_count
log(f"\nDedup result: {new_vectors} new vectors added (ideal: 0)")
log(f"  Before: {before_count}, After: {after_count}")

# ============================================================
# PHASE 4: UPDATE / CORRECTION
# ============================================================
log(f"\n\nPHASE 4: UPDATE / CORRECTION")
log("-" * 50)

# Update: Steven got a new GPU
log("Updating: Steven upgraded from RTX 4090 to RTX 5090...")
r = brain("Steven upgraded his GPU from RTX 4090 to RTX 5090")
log(f"  Agent: {r.get('answer', '')[:120]}")
log(f"  Steps: {r.get('steps')}, Tools: {r.get('tools_called')}")

time.sleep(1)

# Verify the update
r = brain("What GPU does Steven use?")
answer = r.get("answer", "").lower()
has_5090 = "5090" in answer
has_4090 = "4090" in answer
log(f"  Verify: {r.get('answer', '')[:120]}")
log(f"  Has 5090: {has_5090}, Has 4090 (should be gone): {has_4090}")
if has_5090 and not has_4090:
    log(f"  PASS: Correctly updated to 5090")
elif has_5090:
    log(f"  PARTIAL: Has 5090 but old 4090 still present")
else:
    log(f"  FAIL: 5090 not found in answer")

# Revert for other tests
brain("Steven's GPU is actually an RTX 4090, not 5090")
time.sleep(1)

# ============================================================
# PHASE 5: DELETE + GRAPH CLEANUP
# ============================================================
log(f"\n\nPHASE 5: DELETE + GRAPH CLEANUP")
log("-" * 50)

# Check graph for Pixel (cat) before delete
pixel_before = neo4j_query(f"MATCH (n) WHERE n.user_id = $uid AND toLower(n.name) CONTAINS 'pixel' RETURN n.name, n.mentions")
log(f"  Graph 'pixel' before delete: {pixel_before}")

# Delete cat memory
log("  Deleting cat memory...")
r = brain("Delete all memories about Steven's cat Pixel")
log(f"  Agent: {r.get('answer', '')[:120]}")
log(f"  Steps: {r.get('steps')}, Tools: {r.get('tools_called')}")
time.sleep(2)

# Check graph after delete
pixel_after = neo4j_query(f"MATCH (n) WHERE n.user_id = $uid AND toLower(n.name) CONTAINS 'pixel' RETURN n.name, n.mentions")
log(f"  Graph 'pixel' after delete: {pixel_after}")

if not pixel_after:
    log(f"  PASS: Pixel node cleaned from graph")
else:
    mentions = pixel_after[0][1] if len(pixel_after) > 0 and len(pixel_after[0]) > 1 else "?"
    log(f"  INFO: Pixel node still in graph (mentions: {mentions})")

# Verify search returns nothing
r = brain("Does Steven have a cat?")
answer = r.get("answer", "").lower()
no_cat = any(x in answer for x in ["no memor", "no record", "no info", "not found", "don't have", "no stored"])
log(f"  Search after delete: {r.get('answer', '')[:100]}")
log(f"  {'PASS' if no_cat else 'CHECK'}: {'No cat memories found' if no_cat else 'May have stale data'}")

# ============================================================
# PHASE 6: GRAPH QUALITY AUDIT
# ============================================================
log(f"\n\nPHASE 6: GRAPH QUALITY AUDIT")
log("-" * 50)

nodes = neo4j_nodes()
edges = neo4j_edges()
self_refs = neo4j_self_refs()

log(f"  Total nodes: {len(nodes)}")
log(f"  Total edges: {len(edges)}")
log(f"  Self-referential edges: {len(self_refs)}")

if self_refs:
    log(f"  FAIL: Self-refs found:")
    for sr in self_refs:
        log(f"    {sr}")
else:
    log(f"  PASS: No self-referential edges")

# Entity type distribution
type_counts = {}
for node in nodes:
    name = node[0] if node else "?"
    labels = node[1] if len(node) > 1 else []
    for label in labels:
        if label != "__Entity__":
            type_counts[label] = type_counts.get(label, 0) + 1

log(f"\n  Entity type distribution:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    log(f"    {t}: {c}")

concept_count = type_counts.get("concept", 0)
total = sum(type_counts.values())
if total > 0 and concept_count / total > 0.5:
    log(f"  WARNING: >50% of entities are 'concept' ({concept_count}/{total})")
else:
    log(f"  PASS: Good type diversity")

# Hub connectivity
log(f"\n  Hub connectivity:")
for hub in ["steven", "echoes_of_the_fallen"]:
    hub_edges = [e for e in edges if hub in str(e).lower()]
    log(f"    {hub}: {len(hub_edges)} edges")
    if len(hub_edges) < 3:
        log(f"    WARNING: Hub '{hub}' has few connections")

# Sample edges
log(f"\n  Sample edges (first 15):")
for e in edges[:15]:
    log(f"    {e[0]} --{e[1]}--> {e[2]}")

# ============================================================
# PHASE 7: FACT QUALITY CHECK
# ============================================================
log(f"\n\nPHASE 7: FACT QUALITY (Qdrant content)")
log("-" * 50)

qdrant_n, facts = qdrant_count()
log(f"  Total facts in Qdrant: {qdrant_n}")

# Check for self-contained facts (should have subject names)
orphan_facts = []
for fact in facts:
    lower = fact.lower()
    # A good fact should mention a person, project, or entity name
    has_subject = any(name in lower for name in ["steven", "echoes", "mirjam", "alex", "pixel", "tamm", "radiohead"])
    if not has_subject and len(fact) > 20:
        orphan_facts.append(fact)

log(f"  Facts without clear subject: {len(orphan_facts)}")
if orphan_facts:
    for f in orphan_facts[:5]:
        log(f"    - {f}")

# Check for conversational phrasing (should be clean facts)
conv_phrasing = []
for fact in facts:
    lower = fact.lower()
    if any(phrase in lower for phrase in ["remember that", "hey ", "please ", "keep in mind", "don't forget"]):
        conv_phrasing.append(fact)

log(f"  Facts with conversational phrasing: {len(conv_phrasing)}")
if conv_phrasing:
    for f in conv_phrasing[:5]:
        log(f"    - {f}")

if not orphan_facts and not conv_phrasing:
    log(f"  PASS: All facts are clean and self-contained")

# ============================================================
# PHASE 8: EDGE CASES
# ============================================================
log(f"\n\nPHASE 8: EDGE CASES")
log("-" * 50)

# Multi-language
log("  Test: Estonian input")
r = brain("Mäleta, et Steven armastab Eesti mustaleiba")
log(f"    Store: {r.get('answer', '')[:100]}")
time.sleep(1)
r = brain("Does Steven like any Estonian food?")
log(f"    Search: {r.get('answer', '')[:100]}")

# Very short input
log("\n  Test: Very short input")
r = brain("Remember: Steven likes tea")
log(f"    Store: {r.get('answer', '')[:100]}")

# Ambiguous query
log("\n  Test: Ambiguous query")
r = brain("What does Steven do?")
answer = r.get("answer", "")
log(f"    Answer: {answer[:150]}")
# Should mention developer/programming, not just hobbies
has_work = any(x in answer.lower() for x in ["developer", "typescript", "rust", "programming", "code"])
log(f"    Mentions work: {'YES' if has_work else 'NO'}")

# Complex multi-hop
log("\n  Test: Multi-hop reasoning")
r = brain("What game engine does Steven use for Echoes of the Fallen and why did he choose it?")
answer = r.get("answer", "")
log(f"    Answer: {answer[:200]}")
has_ue5 = "unreal" in answer.lower()
has_nanite = "nanite" in answer.lower()
log(f"    UE5: {'YES' if has_ue5 else 'NO'}, Nanite reason: {'YES' if has_nanite else 'NO'}")

# ============================================================
# PHASE 9: FINAL DB STATE
# ============================================================
log(f"\n\nPHASE 9: FINAL DB STATE")
log("-" * 50)

qdrant_n, facts = qdrant_count()
nodes = neo4j_nodes()
edges = neo4j_edges()

log(f"  Qdrant vectors: {qdrant_n}")
log(f"  Neo4j nodes: {len(nodes)}")
log(f"  Neo4j edges: {len(edges)}")

log(f"\n  All Qdrant facts:")
for i, f in enumerate(sorted(facts)):
    log(f"    {i+1:2d}. {f}")

log(f"\n  All Neo4j nodes:")
for n in nodes:
    labels = [l for l in (n[1] or []) if l != "__Entity__"]
    log(f"    {n[0]} ({', '.join(labels)}, mentions={n[2]})")

log(f"\n  All Neo4j edges:")
for e in edges:
    log(f"    {e[0]} --{e[1]}--> {e[2]}")

# ============================================================
# SUMMARY
# ============================================================
log(f"\n\n{'='*70}")
log(f"  SUMMARY")
log(f"{'='*70}")
log(f"  Inserted: {len(inserts)} facts")
log(f"  Qdrant vectors: {qdrant_n}")
log(f"  Neo4j nodes: {len(nodes)}, edges: {len(edges)}")
log(f"  Search quality: {search_pass}/{len(searches)} ({search_pass/len(searches)*100:.0f}%)")
log(f"  Dedup leaks: {new_vectors} new vectors (ideal: 0)")
log(f"  Self-ref edges: {len(self_refs)}")
log(f"  Orphan facts: {len(orphan_facts)}")
log(f"  Conversational phrasing: {len(conv_phrasing)}")

# Write log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\n  Log saved to: {LOG_FILE}")
