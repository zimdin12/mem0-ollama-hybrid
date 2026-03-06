"""Comprehensive test suite for OpenMemory system."""
import requests
import time
import json
import sys

BASE = "http://localhost:8765"


def safe_print(s):
    """Print with unicode safety for Windows."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode())


def test_bulk_insertion():
    """Test 1: Multi-domain bulk insertion with deep entity chains."""
    print("=" * 60)
    print("TEST 1: Multi-domain bulk insertion")
    print("=" * 60)

    batches = {
        "Personal": (
            "Steven is a PHP developer from Tallinn, Estonia who started programming at age 14.\n"
            "Steven is learning C++ specifically for building Echoes of the Fallen.\n"
            "Steven prefers dark mode in all applications and editors.\n"
            "Steven prefers local-first AI solutions without cloud API dependencies.\n"
            "Steven uses Docker with WSL2 on Windows 11 for all development work.\n"
            "Steven uses vim as his primary text editor with custom keybindings.\n"
            "Steven has an NVIDIA RTX 4090 GPU with 24GB VRAM in his workstation.\n"
            "Steven enjoys cooking as a hobby, especially baking sourdough bread.\n"
            "Alex is Steven's friend who works at Google on ML infrastructure team.\n"
            "Tom is Steven's brother who works as a senior designer at Figma.\n"
            "Steven uses Proxmox for home server virtualization with ZFS storage.\n"
            "Steven monitors his infrastructure with Grafana dashboards and Prometheus metrics."
        ),
        "Echoes (deep chain)": (
            "Echoes of the Fallen is a voxel-based roguelike exploration game developed by Steven.\n"
            "Echoes of the Fallen uses C++ as its primary programming language.\n"
            "Echoes of the Fallen uses dual contouring algorithm for terrain mesh generation.\n"
            "Echoes of the Fallen has a custom voxel renderer built on Vulkan graphics API.\n"
            "The voxel renderer in Echoes of the Fallen uses frustum culling for performance.\n"
            "Echoes of the Fallen implements a chunk-based world system with 32x32x32 voxel chunks.\n"
            "Echoes of the Fallen stores chunk data in SQLite databases using WAL mode for concurrency.\n"
            "Echoes of the Fallen features water magic, fire spells, and earth manipulation as core abilities.\n"
            "Echoes of the Fallen has a non-linear skill tree with 47 unique abilities.\n"
            "Echoes of the Fallen targets 4-player online co-op with peer-to-peer networking.\n"
            "Echoes of the Fallen uses the Opus audio codec for voice chat between players."
        ),
        "Infrastructure": (
            "OpenClaw runs on port 3000 as a gateway and control UI inside Docker.\n"
            "Ollama runs inside Docker on port 11435 serving local LLMs for the memory system.\n"
            "The memory system uses qwen3:4b model for fact extraction and review.\n"
            "The memory system uses qwen3-embedding:0.6b model for generating 1024-dimensional vectors.\n"
            "Qdrant vector database stores the embeddings on port 6333 with cosine similarity.\n"
            "Neo4j graph database stores entity relationships with APOC plugin enabled.\n"
            "OpenMemory API on port 8765 combines vector, graph, and temporal search.\n"
            "OpenMemory exposes tools via MCP protocol over SSE transport."
        ),
        "Music (curveball domain)": (
            "Steven has been learning music production using Ableton Live for the past 6 months.\n"
            "Steven uses a Focusrite Scarlett 2i2 audio interface for recording.\n"
            "Steven is learning music theory through the Hooktheory website.\n"
            "Steven wants to compose the Echoes of the Fallen soundtrack himself using orchestral samples.\n"
            "Steven bought the Spitfire Audio BBC Symphony Orchestra library for cinematic sound."
        ),
    }

    total_stored = 0
    for name, text in batches.items():
        r = requests.post(
            f"{BASE}/api/v1/memories/",
            json={"user_id": "steven", "text": text, "app": "claude-code"},
        )
        d = r.json()
        count = d.get("count", len(d.get("results", [])))
        total_stored += count
        print(f"  {name}: {count} stored")

    print(f"  TOTAL: {total_stored} facts stored")
    print("  Waiting 15s for graph extraction...")
    time.sleep(15)
    return total_stored


def test_conversation_memory():
    """Test 2: Conversation memory with curveballs and topic mixing."""
    print("\n" + "=" * 60)
    print("TEST 2: Conversation memory with curveballs")
    print("=" * 60)

    turns = [
        # Two projects in one message
        (
            "I am also building a cooking recipe app called ChefMate using Laravel and PostgreSQL. It is separate from the game.",
            "Interesting! ChefMate with Laravel and PostgreSQL is a solid stack.",
        ),
        # Correction + both projects referenced
        (
            "Actually ChefMate will use MySQL not PostgreSQL, I changed my mind. For Echoes I am switching from peer-to-peer to dedicated servers using Photon.",
            "MySQL is fine for ChefMate. Photon is great for dedicated game servers.",
        ),
        # Ambiguous 'it' - which project?
        (
            "The app needs user authentication with OAuth2 and social login. I want to use Redis for session caching.",
            "OAuth2 with social login is standard. Redis for sessions is fast.",
        ),
        # Deep technical detail
        (
            "For the dual contouring in Echoes I implemented Hermite data interpolation using the QEF minimizer from Ju et al. 2002 paper.",
            "The QEF minimizer from Ju et al. is the gold standard for dual contouring.",
        ),
        # Preference change
        (
            "Actually I switched from vim to Neovim last week. The Lua config is so much better. I use LazyVim distribution.",
            "Neovim with LazyVim is excellent. The Lua ecosystem is much more modern.",
        ),
    ]

    total_stored = 0
    for i, (u, a) in enumerate(turns):
        r = requests.post(
            f"{BASE}/api/v1/memories/conversation",
            json={
                "user_id": "steven",
                "user_message": u,
                "llm_response": a,
                "app": "claude-code",
            },
        )
        d = r.json()
        stored = d.get("facts_stored", 0)
        total_stored += stored
        facts = d.get("extracted_facts", [])
        print(f"  Turn {i+1}: stored={stored}/{d.get('facts_extracted', 0)}")
        for f in facts:
            safe_print(f"    -> {f[:100]}")
        time.sleep(1)

    print(f"  TOTAL: {total_stored} facts stored from conversation")
    print("  Waiting 15s for graph extraction...")
    time.sleep(15)
    return total_stored


def test_search_quality():
    """Test 3: Search quality across domains."""
    print("\n" + "=" * 60)
    print("TEST 3: Search quality")
    print("=" * 60)

    queries = [
        "steven tools and preferences",
        "echoes of the fallen technology stack",
        "cooking hobby",
        "tom alex people steven knows",
        "chefmate project",
        "music production",
        "dual contouring implementation",
        "neovim editor",
        "infrastructure monitoring",
        "vulkan renderer",
    ]

    for q in queries:
        r = requests.post(
            f"{BASE}/api/v1/memories/search",
            json={"query": q, "user_id": "steven", "limit": 5},
        )
        results = r.json().get("results", [])
        sources = [res.get("source", "?") for res in results]
        source_summary = ", ".join(sorted(set(sources)))
        print(f'  "{q}"')
        print(f"    {len(results)} results, sources: [{source_summary}]")
        if results:
            top = results[0]
            safe_print(
                f"    top: [{top.get('source')}:{top.get('score', 0):.2f}] "
                f"{top.get('memory', '')[:90]}"
            )


def test_dedup():
    """Test 4: Deduplication - inserting same facts again should add 0."""
    print("\n" + "=" * 60)
    print("TEST 4: Deduplication")
    print("=" * 60)

    text = (
        "Steven is a PHP developer from Tallinn, Estonia.\n"
        "Steven uses vim as his primary text editor.\n"
        "Echoes of the Fallen is a voxel-based roguelike game."
    )
    r = requests.post(
        f"{BASE}/api/v1/memories/",
        json={"user_id": "steven", "text": text, "app": "claude-code"},
    )
    d = r.json()
    count = d.get("count", len(d.get("results", [])))
    status = "PASS" if count == 0 else "WARN"
    print(f"  Re-inserted 3 existing facts: {count} new stored [{status}]")


def test_graph_quality():
    """Test 5: Graph database audit."""
    print("\n" + "=" * 60)
    print("TEST 5: Graph quality audit")
    print("=" * 60)

    auth = ("neo4j", "openmemory")
    neo = "http://localhost:8474"

    # Entity type distribution
    r = requests.post(
        f"{neo}/db/neo4j/tx/commit",
        auth=auth,
        json={
            "statements": [
                {
                    "statement": "MATCH (n) RETURN labels(n)[0] AS type, count(*) AS cnt ORDER BY cnt DESC"
                }
            ]
        },
    )
    print("  Entity type distribution:")
    total = 0
    types_found = []
    for row in r.json()["results"][0]["data"]:
        t, c = row["row"]
        total += c
        types_found.append(t)
        print(f"    {t}: {c}")
    print(f"    TOTAL: {total}")

    # Check for concept/metric types (should be 0)
    bad_types = [t for t in types_found if t in ("concept", "metric")]
    if bad_types:
        print(f"  WARNING: Found blocked types: {bad_types}")
    else:
        print("  PASS: No blocked types (concept/metric)")

    # Self-referential edges
    r = requests.post(
        f"{neo}/db/neo4j/tx/commit",
        auth=auth,
        json={
            "statements": [
                {"statement": "MATCH (n)-[r]->(n) RETURN n.name, type(r)"}
            ]
        },
    )
    self_refs = r.json()["results"][0]["data"]
    if self_refs:
        print(f"  FAIL: {len(self_refs)} self-referential edges!")
        for row in self_refs:
            print(f"    {row['row']}")
    else:
        print("  PASS: 0 self-referential edges")

    # Hub connectivity
    r = requests.post(
        f"{neo}/db/neo4j/tx/commit",
        auth=auth,
        json={
            "statements": [
                {
                    "statement": (
                        "MATCH (n)-[r]->() "
                        "WHERE labels(n)[0] IN ['person', 'project'] "
                        "RETURN n.name, labels(n)[0], count(r) AS rels "
                        "ORDER BY rels DESC"
                    )
                }
            ]
        },
    )
    print("  Hub connectivity:")
    for row in r.json()["results"][0]["data"]:
        name, ntype, rels = row["row"]
        print(f"    {name} ({ntype}): {rels} outgoing relationships")

    # Non-hub sources (should be 0)
    r = requests.post(
        f"{neo}/db/neo4j/tx/commit",
        auth=auth,
        json={
            "statements": [
                {
                    "statement": (
                        "MATCH (a)-[r]->(b) "
                        "WHERE NOT labels(a)[0] IN ['person', 'project'] "
                        "RETURN a.name, labels(a)[0], type(r), b.name "
                        "LIMIT 10"
                    )
                }
            ]
        },
    )
    non_hub = r.json()["results"][0]["data"]
    if non_hub:
        print(f"  WARNING: {len(non_hub)} relationships from non-hub sources:")
        for row in non_hub:
            print(f"    [{row['row'][1]}] {row['row'][0]} --{row['row'][2]}--> {row['row'][3]}")
    else:
        print("  PASS: All relationships flow from person/project hubs")

    # All relationships
    r = requests.post(
        f"{neo}/db/neo4j/tx/commit",
        auth=auth,
        json={
            "statements": [
                {
                    "statement": (
                        "MATCH (a)-[r]->(b) "
                        "RETURN labels(a)[0], a.name, type(r), b.name, labels(b)[0] "
                        "ORDER BY a.name, type(r)"
                    )
                }
            ]
        },
    )
    rels = r.json()["results"][0]["data"]
    print(f"\n  All relationships ({len(rels)}):")
    for row in rels:
        st, s, rel, t, tt = row["row"]
        print(f"    [{st}] {s} --{rel}--> {t} [{tt}]")


def test_qdrant_count():
    """Test 6: Qdrant vector count."""
    print("\n" + "=" * 60)
    print("TEST 6: Qdrant vector store")
    print("=" * 60)
    r = requests.post(
        "http://localhost:6333/collections/openmemory/points/count",
        json={"exact": True},
    )
    count = r.json().get("result", {}).get("count", 0)
    print(f"  Total vectors: {count}")


if __name__ == "__main__":
    test_bulk_insertion()
    test_conversation_memory()
    test_search_quality()
    test_dedup()
    test_graph_quality()
    test_qdrant_count()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
