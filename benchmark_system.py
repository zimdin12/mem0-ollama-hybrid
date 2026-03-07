"""
Comprehensive end-to-end system benchmark.

Tests the FULL pipeline: API -> extraction -> storage -> search -> dedup -> graph.
One model at a time, with VRAM cleanup and model switching via config API.

Usage: docker cp benchmark_system.py openmemory-mcp:/usr/src/openmemory/ && \
       MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/benchmark_system.py [--model MODEL]
"""
import requests
import json
import time
import sys
import os

API_URL = "http://localhost:8765/api"
OLLAMA_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
USER_ID = "steven"
APP_NAME = "openmemory"

MODELS = [
    ('ministral-3:3b', {'temperature': 0.05, 'top_p': 0.8, 'top_k': 40}),
    ('qwen3.5:4b', {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}),
    ('qwen3.5:9b', {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}),
    ('qwen3:4b-instruct-2507-q4_K_M', {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}),
]

# ============================================================================
# Test data — diverse types
# ============================================================================

# Phase 1: Single facts (should each create exactly 1 memory)
SINGLE_FACTS = [
    "Steven is a PHP developer.",
    "Steven lives in Tallinn, Estonia.",
    "Steven prefers dark mode in all editors.",
    "Steven's favorite programming language is Rust.",
    "Steven does not use Java.",
]

# Phase 2: Dense chunk (should create many facts + graph nodes)
DENSE_CHUNK = """Steven built Echoes of the Fallen, a voxel-based roguelike exploration game.
The game is developed with Godot Engine using C++ and GDScript.
It features procedural terrain generation using dual contouring and marching cubes algorithms.
The world is a grim, nature-reclaimed post-apocalyptic setting.
Players acquire permanent knowledge while items are temporary.
Total development time is 18 months as a solo developer.
The game targets 60fps at 4K resolution on modern hardware."""

# Phase 3: Multi-person relationships
MULTI_PERSON = """Jake is the lead DevOps engineer at CloudTech Inc.
Maria is a frontend developer who works with Jake on the deployment pipeline.
Steven previously worked at Acme Corp as a senior PHP developer but quit in 2024.
Steven now collaborates with Jake on open-source projects."""

# Phase 4: Technical stack (impersonal — tests user entity injection)
TECH_STACK = """The project uses PostgreSQL with pgvector for vector embeddings.
Redis handles caching and session management.
Nginx serves as the reverse proxy with SSL termination.
The API is built with FastAPI on Python 3.12.
Docker Compose orchestrates all 6 services.
CI/CD runs on GitHub Actions with pytest and mypy checks."""

# Phase 5: Conversation (user + assistant messages)
CONVERSATION = {
    "messages": [
        {"role": "user", "content": "I've been thinking about switching from Neovim to VS Code for my Godot development."},
        {"role": "assistant", "content": "That's a common consideration. VS Code has excellent Godot integration with the godot-tools extension. What's making you consider the switch?"},
        {"role": "user", "content": "Mainly the debugging support. Neovim's DAP adapter for Godot is unreliable. But I'll keep Neovim for everything else — I just love modal editing too much."},
    ]
}

# Phase 6: Noise (should NOT create any memories)
NOISE_INPUTS = [
    "Hey, how are you doing today?",
    "Can you search for Python tutorials?",
    "Thanks, that was helpful!",
    "What time is it?",
    "Show me the code for the login page.",
]

# Phase 7: Negations and temporal
NEGATION_TEMPORAL = """Steven stopped using Windows in 2023 and switched to Linux.
He rejected React in favor of Svelte for frontend work.
Steven never uses tabs — always spaces.
He previously preferred PostgreSQL but now uses SQLite for small projects."""

# Phase 8: Dedup test — same content again
DEDUP_CONTENT = "Steven is a PHP developer who lives in Tallinn, Estonia."

# Phase 9: Contradictions
CONTRADICTION = "Steven now lives in Helsinki, Finland. He switched from PHP to Go as his main language."

# Phase 10: Estonian mixed
ESTONIAN_MIX = "Steven elab Tallinnas ja töötab IT-sektoris. He builds games with Godot Engine in his spare time."

# Phase 11: Chained facts (should create connected graph)
CHAINED_FACTS = [
    "Steven is building a memory system called OpenMemory.",
    "OpenMemory uses Qdrant as its vector database.",
    "OpenMemory also uses Neo4j for graph storage.",
    "Neo4j stores entity relationships extracted by Ollama.",
    "Ollama runs on Steven's RTX 4090 GPU.",
]

# Phase 12: Very short facts
SHORT_FACTS = [
    "Steven likes coffee.",
    "Steven has a cat named Luna.",
    "Steven speaks Estonian and English.",
]

# Phase 13: Complex technical paragraph
COMPLEX_TECH = """The rendering pipeline in Echoes of the Fallen uses a deferred shading approach
with screen-space ambient occlusion (SSAO) and temporal anti-aliasing (TAA). The voxel engine
implements a custom octree structure for LOD management, allowing seamless transitions between
detail levels. Chunk meshing is parallelized across 8 threads using Godot's WorkerThreadPool,
achieving sub-16ms frame times even with 256x256x64 chunk dimensions. The dual contouring
algorithm preserves sharp features that marching cubes would smooth over, critical for the
architectural ruins that define the game's visual identity."""


# ============================================================================
# Helper functions
# ============================================================================

def api_add(text, timeout=30):
    """Add memories via REST API."""
    resp = requests.post(f"{API_URL}/v1/memories/", json={
        "text": text,
        "user_id": USER_ID,
        "app_id": APP_NAME,
        "infer": True,
    }, timeout=timeout)
    return resp

def api_search(query, limit=10, offset=0):
    """Search memories via REST API."""
    resp = requests.post(f"{API_URL}/v1/memories/search", json={
        "query": query,
        "user_id": USER_ID,
        "app_id": APP_NAME,
        "limit": limit,
        "offset": offset,
    }, timeout=30)
    return resp

def api_conversation(user_msg, llm_response, timeout=30):
    """Process conversation via REST API."""
    resp = requests.post(f"{API_URL}/v1/memories/conversation", json={
        "user_message": user_msg,
        "llm_response": llm_response,
        "user_id": USER_ID,
    }, timeout=timeout)
    return resp

def api_list(page_size=1):
    """List all memories (just to get total count)."""
    resp = requests.get(f"{API_URL}/v1/memories/", params={
        "user_id": USER_ID,
        "size": page_size,
        "page": 1,
    }, timeout=30)
    return resp

def api_delete_all():
    """Delete all memories for user via delete_all endpoint."""
    resp = requests.post(f"{API_URL}/v1/memories/delete_all", json={
        "user_id": USER_ID,
    }, timeout=120)
    return resp

def api_graph_stats():
    """Get graph statistics."""
    resp = requests.get(f"{API_URL}/v1/memories/graph/stats", params={
        "user_id": USER_ID,
    }, timeout=60)
    return resp

def api_graph_context(topic):
    """Get graph context for a topic (entity traversal, no memories — MENTIONED_IN not implemented)."""
    resp = requests.get(f"{API_URL}/v1/memories/graph/context/{topic}", params={
        "user_id": USER_ID,
        "include_memories": "false",
    }, timeout=30)
    return resp

def switch_model(model_name):
    """Switch the extraction LLM via config API. Returns True on success."""
    try:
        resp = requests.put(f"{API_URL}/v1/config/mem0/llm", json={
            "provider": "ollama",
            "config": {
                "model": model_name,
                "temperature": 0,
                "max_tokens": 4000,
                "ollama_base_url": OLLAMA_URL,
            }
        }, timeout=10)
        if resp.status_code == 200:
            # Verify
            r2 = requests.get(f"{API_URL}/v1/config/mem0/llm", timeout=5)
            actual = r2.json().get('config', {}).get('model', '?')
            return actual == model_name
        return False
    except Exception as e:
        print(f"    [switch_model error: {e}]", flush=True)
        return False

def unload_model(model):
    """Unload model from VRAM."""
    try:
        requests.post(f'{OLLAMA_URL}/api/chat', json={
            'model': model, 'messages': [{'role': 'user', 'content': 'x'}],
            'stream': False, 'think': False,
            'options': {'num_predict': 1},
            'keep_alive': 0,
        }, timeout=30)
    except:
        pass

def unload_all():
    """Unload all models."""
    for m, _ in MODELS:
        unload_model(m)

def count_memories():
    """Count total memories via list endpoint."""
    try:
        resp = api_list(1)
        if resp.status_code == 200:
            return resp.json().get('total', 0)
    except Exception as e:
        print(f"    [count error: {e}]", flush=True)
    return 0

def list_memory_contents(limit=200):
    """Get all memory texts for quality inspection."""
    try:
        resp = requests.get(f"{API_URL}/v1/memories/", params={
            "user_id": USER_ID,
            "size": limit,
            "page": 1,
        }, timeout=30)
        if resp.status_code == 200:
            items = resp.json().get('items', [])
            return [item.get('content', item.get('memory', '')) for item in items]
    except:
        pass
    return []


# ============================================================================
# Test runner
# ============================================================================

def run_test(model_name, model_config):
    print(f"\n{'='*80}", flush=True)
    print(f"SYSTEM TEST: {model_name}", flush=True)
    print(f"Config: {model_config}", flush=True)
    print(f"{'='*80}", flush=True)

    # Switch model via config API
    print(f"\n  Switching model to {model_name}...", end=' ', flush=True)
    if switch_model(model_name):
        print("OK", flush=True)
    else:
        print("FAILED — skipping this model", flush=True)
        return None

    results = {
        'model': model_name,
        'phases': {},
        'issues': [],
        'quality': [],
        'total_time': 0,
    }
    test_start = time.time()

    # --- Phase 0: Clean slate ---
    print("\n  [Phase 0] Cleanup...", end=' ', flush=True)
    api_delete_all()
    time.sleep(3)
    initial_count = count_memories()
    if initial_count > 0:
        print(f"WARNING: {initial_count} memories remain after delete_all", flush=True)
        results['issues'].append(f"Phase 0: {initial_count} memories after cleanup")
    else:
        print("done (clean)", flush=True)

    # --- Phase 1: Single facts ---
    print("\n  [Phase 1] Single facts (5 inserts)...", flush=True)
    p1_start = time.time()
    for i, fact in enumerate(SINGLE_FACTS):
        t0 = time.time()
        resp = api_add(fact, timeout=60)
        dt = time.time() - t0
        status = "OK" if resp.status_code == 200 else f"ERR:{resp.status_code}"
        print(f"    {i+1}. [{status}] {dt:.1f}s — {fact[:60]}", flush=True)
    p1_time = time.time() - p1_start
    time.sleep(3)
    p1_count = count_memories()
    print(f"    => {p1_count} memories, {p1_time:.1f}s total", flush=True)
    results['phases']['single_facts'] = {'count': p1_count, 'time': p1_time}
    if p1_count < 4:
        results['issues'].append(f"Phase 1: only {p1_count} memories from 5 facts")

    # --- Phase 2: Dense chunk ---
    print("\n  [Phase 2] Dense chunk (Echoes project)...", flush=True)
    p2_start = time.time()
    resp = api_add(DENSE_CHUNK, timeout=120)
    p2_time = time.time() - p2_start
    time.sleep(5)
    p2_count = count_memories()
    p2_new = p2_count - p1_count
    print(f"    => +{p2_new} new memories ({p2_count} total), {p2_time:.1f}s", flush=True)
    results['phases']['dense_chunk'] = {'count': p2_new, 'time': p2_time}
    if p2_new < 4:
        results['issues'].append(f"Phase 2: only {p2_new} memories from dense chunk")

    # --- Phase 3: Multi-person ---
    print("\n  [Phase 3] Multi-person relationships...", flush=True)
    p3_start = time.time()
    before = count_memories()
    resp = api_add(MULTI_PERSON, timeout=120)
    p3_time = time.time() - p3_start
    time.sleep(5)
    p3_new = count_memories() - before
    print(f"    => +{p3_new} new memories, {p3_time:.1f}s", flush=True)
    results['phases']['multi_person'] = {'count': p3_new, 'time': p3_time}

    # --- Phase 4: Tech stack ---
    print("\n  [Phase 4] Technical stack (impersonal text)...", flush=True)
    p4_start = time.time()
    before = count_memories()
    resp = api_add(TECH_STACK, timeout=120)
    p4_time = time.time() - p4_start
    time.sleep(5)
    p4_new = count_memories() - before
    print(f"    => +{p4_new} new memories, {p4_time:.1f}s", flush=True)
    results['phases']['tech_stack'] = {'count': p4_new, 'time': p4_time}

    # --- Phase 5: Conversation ---
    print("\n  [Phase 5] Conversation (3 turns)...", flush=True)
    p5_start = time.time()
    before = count_memories()
    for i in range(0, len(CONVERSATION["messages"]) - 1, 2):
        user_msg = CONVERSATION["messages"][i]["content"]
        llm_msg = CONVERSATION["messages"][i + 1]["content"] if i + 1 < len(CONVERSATION["messages"]) else ""
        resp = api_conversation(user_msg, llm_msg, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            print(f"    Turn {i//2+1}: {data.get('facts_stored', 0)} stored, {data.get('duplicates_skipped', 0)} deduped", flush=True)
        else:
            print(f"    Turn {i//2+1}: HTTP {resp.status_code}", flush=True)
    if len(CONVERSATION["messages"]) % 2 == 1:
        last_msg = CONVERSATION["messages"][-1]["content"]
        resp = api_conversation(last_msg, "", timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            print(f"    Turn {len(CONVERSATION['messages'])//2+1}: {data.get('facts_stored', 0)} stored", flush=True)
    p5_time = time.time() - p5_start
    time.sleep(3)
    p5_new = count_memories() - before
    print(f"    => +{p5_new} new memories, {p5_time:.1f}s", flush=True)
    results['phases']['conversation'] = {'count': p5_new, 'time': p5_time}

    # --- Phase 6: Noise rejection ---
    print("\n  [Phase 6] Noise rejection (5 inputs)...", flush=True)
    p6_start = time.time()
    before = count_memories()
    for noise in NOISE_INPUTS:
        api_add(noise, timeout=30)
    time.sleep(2)
    noise_leaked = count_memories() - before
    p6_time = time.time() - p6_start
    print(f"    => {noise_leaked} leaked (should be 0), {p6_time:.1f}s", flush=True)
    results['phases']['noise'] = {'leaked': noise_leaked, 'time': p6_time}
    if noise_leaked > 0:
        results['issues'].append(f"Phase 6: {noise_leaked} noise inputs leaked")

    # --- Phase 7: Negation/temporal ---
    print("\n  [Phase 7] Negation & temporal...", flush=True)
    p7_start = time.time()
    before = count_memories()
    resp = api_add(NEGATION_TEMPORAL, timeout=120)
    p7_time = time.time() - p7_start
    time.sleep(3)
    p7_new = count_memories() - before
    print(f"    => +{p7_new} new memories, {p7_time:.1f}s", flush=True)
    results['phases']['negation_temporal'] = {'count': p7_new, 'time': p7_time}

    # --- Phase 8: Dedup test ---
    print("\n  [Phase 8] Dedup (re-insert known facts)...", flush=True)
    before = count_memories()
    p8_start = time.time()
    resp = api_add(DEDUP_CONTENT, timeout=60)
    p8_time = time.time() - p8_start
    time.sleep(2)
    new_from_dedup = count_memories() - before
    print(f"    => +{new_from_dedup} new (should be 0), {p8_time:.1f}s", flush=True)
    results['phases']['dedup'] = {'new': new_from_dedup, 'time': p8_time}
    if new_from_dedup > 1:
        results['issues'].append(f"Phase 8: dedup failed — {new_from_dedup} duplicates added")

    # --- Phase 9: Contradiction ---
    print("\n  [Phase 9] Contradiction (updated location + language)...", flush=True)
    p9_start = time.time()
    before = count_memories()
    resp = api_add(CONTRADICTION, timeout=60)
    p9_time = time.time() - p9_start
    time.sleep(3)
    p9_new = count_memories() - before
    print(f"    => +{p9_new} new memories, {p9_time:.1f}s", flush=True)
    results['phases']['contradiction'] = {'count': p9_new, 'time': p9_time}

    # --- Phase 10: Estonian mix ---
    print("\n  [Phase 10] Estonian mixed language...", flush=True)
    p10_start = time.time()
    before = count_memories()
    resp = api_add(ESTONIAN_MIX, timeout=60)
    p10_time = time.time() - p10_start
    time.sleep(3)
    p10_new = count_memories() - before
    print(f"    => +{p10_new} new, {p10_time:.1f}s", flush=True)
    results['phases']['estonian'] = {'count': p10_new, 'time': p10_time}

    # --- Phase 11: Chained facts ---
    print("\n  [Phase 11] Chained facts (5 connected inserts)...", flush=True)
    p11_start = time.time()
    before = count_memories()
    for i, fact in enumerate(CHAINED_FACTS):
        t0 = time.time()
        resp = api_add(fact, timeout=60)
        dt = time.time() - t0
        print(f"    {i+1}. {dt:.1f}s — {fact[:60]}", flush=True)
    p11_time = time.time() - p11_start
    time.sleep(5)
    p11_new = count_memories() - before
    print(f"    => +{p11_new} new memories, {p11_time:.1f}s", flush=True)
    results['phases']['chained'] = {'count': p11_new, 'time': p11_time}

    # --- Phase 12: Short facts ---
    print("\n  [Phase 12] Very short facts (3 inserts)...", flush=True)
    p12_start = time.time()
    before = count_memories()
    for fact in SHORT_FACTS:
        api_add(fact, timeout=60)
    p12_time = time.time() - p12_start
    time.sleep(3)
    p12_new = count_memories() - before
    print(f"    => +{p12_new} new memories, {p12_time:.1f}s", flush=True)
    results['phases']['short_facts'] = {'count': p12_new, 'time': p12_time}

    # --- Phase 13: Complex technical paragraph ---
    print("\n  [Phase 13] Complex technical paragraph...", flush=True)
    p13_start = time.time()
    before = count_memories()
    resp = api_add(COMPLEX_TECH, timeout=120)
    p13_time = time.time() - p13_start
    time.sleep(5)
    p13_new = count_memories() - before
    print(f"    => +{p13_new} new memories, {p13_time:.1f}s", flush=True)
    results['phases']['complex_tech'] = {'count': p13_new, 'time': p13_time}

    total_memories = count_memories()
    insert_time = time.time() - test_start

    # ====================================================================
    # Quality inspection — check stored memories for issues
    # ====================================================================
    print(f"\n  [Quality Check] Inspecting {total_memories} stored memories...", flush=True)
    all_memories = list_memory_contents(200)

    # Check for first-person leaks
    first_person = [m for m in all_memories if any(
        w in m.lower().split() for w in ['i ', ' i ', 'my ', ' my ']
    ) and 'steven' not in m.lower()]
    # More precise check — word boundary "I" or "my" at start
    first_person = []
    for m in all_memories:
        words = m.split()
        if any(w in ('I', 'My', 'my') for w in words) and 'steven' not in m.lower():
            first_person.append(m)
    if first_person:
        print(f"    ISSUE: {len(first_person)} memories with first-person (I/My) instead of Steven:", flush=True)
        for m in first_person[:3]:
            print(f"      - {m[:80]}", flush=True)
        results['quality'].append(f"{len(first_person)} first-person leaks")

    # Check for "User" references
    user_refs = [m for m in all_memories if 'the user' in m.lower() or m.startswith('User ')]
    if user_refs:
        print(f"    ISSUE: {len(user_refs)} memories say 'User' instead of 'Steven':", flush=True)
        for m in user_refs[:3]:
            print(f"      - {m[:80]}", flush=True)
        results['quality'].append(f"{len(user_refs)} 'User' references")

    # Check for context-stripped facts (orphaned facts without subject)
    orphaned = []
    for m in all_memories:
        # Facts that start with generic words without a proper noun
        first_word = m.split()[0] if m.split() else ""
        if first_word.lower() in ('uses', 'has', 'is', 'was', 'does', 'can', 'will', 'the', 'a', 'total', 'it'):
            orphaned.append(m)
    if orphaned:
        print(f"    ISSUE: {len(orphaned)} potentially context-stripped facts:", flush=True)
        for m in orphaned[:3]:
            print(f"      - {m[:80]}", flush=True)
        results['quality'].append(f"{len(orphaned)} context-stripped facts")

    if not first_person and not user_refs and not orphaned:
        print(f"    All {total_memories} memories passed quality checks", flush=True)

    # ====================================================================
    # Search tests
    # ====================================================================
    print(f"\n  [Search Tests]", flush=True)
    searches = {
        'PHP developer': ['php', 'developer'],
        'Echoes of the Fallen': ['echoes', 'fallen'],
        'Godot Engine': ['godot'],
        'Jake DevOps': ['jake', 'devops'],
        'PostgreSQL': ['postgresql', 'postgres', 'pgvector'],
        'dark mode': ['dark'],
        'does not use Java': ['java'],
        'lives in': ['tallinn', 'helsinki', 'finland'],
        'Neovim VS Code': ['neovim', 'code'],
        'OpenMemory system': ['openmemory', 'qdrant'],
        'cat named Luna': ['luna', 'cat'],
        'rendering pipeline': ['deferred', 'ssao', 'octree'],
        'stopped using Windows': ['windows', 'linux', 'stopped'],
    }
    search_pass = 0
    search_fail = 0
    search_times = []
    for query, expected_kws in searches.items():
        t0 = time.time()
        resp = api_search(query)
        dt = time.time() - t0
        search_times.append(dt)
        if resp.status_code == 200:
            data = resp.json()
            results_list = data.get('results', [])
            all_text = ' '.join([str(r) for r in results_list]).lower()
            found = any(kw.lower() in all_text for kw in expected_kws)
            status = "PASS" if found else "MISS"
            if found:
                search_pass += 1
            else:
                search_fail += 1
                results['issues'].append(f"Search '{query}': no results matching {expected_kws}")
            print(f"    [{status}] '{query}' -> {len(results_list)} results, {dt*1000:.0f}ms", flush=True)
        else:
            search_fail += 1
            print(f"    [ERR] '{query}' -> HTTP {resp.status_code}, {dt*1000:.0f}ms", flush=True)
    avg_search = sum(search_times) / len(search_times) * 1000 if search_times else 0
    results['phases']['search'] = {'pass': search_pass, 'fail': search_fail, 'avg_ms': round(avg_search)}

    # ====================================================================
    # Graph tests
    # ====================================================================
    print(f"\n  [Graph Tests]", flush=True)
    try:
        resp = api_graph_stats()
        if resp.status_code == 200:
            stats = resp.json()
            nodes = stats.get('total_entities', stats.get('total_nodes', 0))
            edges = stats.get('total_relationships', stats.get('total_edges', 0))
            entity_types = stats.get('entity_types', [])
            print(f"    Nodes: {nodes}, Edges: {edges}", flush=True)
            if entity_types:
                type_str = ', '.join(f"{t['type']}:{t['count']}" for t in entity_types[:10])
                print(f"    Types: {type_str}", flush=True)
            results['phases']['graph'] = {'nodes': nodes, 'edges': edges}

            # Check entity type diversity
            if entity_types:
                concept_count = sum(t['count'] for t in entity_types if t['type'].lower() == 'concept')
                if concept_count > nodes * 0.7:
                    results['issues'].append(f"Graph: {concept_count}/{nodes} nodes are 'concept' (>70%)")

            if nodes < 10:
                results['issues'].append(f"Graph: only {nodes} nodes (expected 15+)")
            if edges < 10:
                results['issues'].append(f"Graph: only {edges} edges (expected 15+)")
        else:
            print(f"    Graph stats: HTTP {resp.status_code}", flush=True)
    except Exception as e:
        print(f"    Graph stats error: {e}", flush=True)

    # Graph context queries (entity traversal only — MENTIONED_IN not implemented)
    for topic in ['steven', 'echoes_of_the_fallen', 'openmemory']:
        try:
            resp = api_graph_context(topic)
            if resp.status_code == 200:
                data = resp.json()
                root = data.get('root_entities', [])
                related = data.get('related_entities', {})
                total_related = sum(len(v) for v in related.values())
                print(f"    Context '{topic}': {len(root)} root, {total_related} related entities", flush=True)
            else:
                print(f"    Context '{topic}': HTTP {resp.status_code}", flush=True)
        except Exception as e:
            print(f"    Context '{topic}': error {e}", flush=True)

    # Self-referential edge check
    try:
        from neo4j import GraphDatabase
        neo4j_url = os.environ.get('NEO4J_URL', 'bolt://neo4j:7687')
        neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        neo4j_pass = os.environ.get('NEO4J_PASSWORD', 'openmemory')
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pass))
        with driver.session() as session:
            self_refs = session.run(
                "MATCH (n)-[r]->(n) WHERE n.user_id = $uid RETURN n.name, type(r) LIMIT 10",
                uid=USER_ID
            ).data()
            if self_refs:
                print(f"    ISSUE: {len(self_refs)} self-referential edges found:", flush=True)
                for sr in self_refs[:5]:
                    print(f"      {sr['n.name']} --{sr['type(r)']}--> itself", flush=True)
                results['issues'].append(f"Graph: {len(self_refs)} self-referential edges")
            else:
                print(f"    No self-referential edges (good)", flush=True)
        driver.close()
    except Exception as e:
        print(f"    Self-ref check: {e}", flush=True)

    # ====================================================================
    # Pagination test
    # ====================================================================
    print(f"\n  [Pagination Test]", flush=True)
    resp1 = api_search("Steven", limit=5, offset=0)
    resp2 = api_search("Steven", limit=5, offset=5)
    if resp1.status_code == 200 and resp2.status_code == 200:
        r1 = resp1.json().get('results', [])
        r2 = resp2.json().get('results', [])
        # Compare by memory content to check overlap
        r1_contents = set(str(r.get('memory', r.get('content', ''))) for r in r1)
        r2_contents = set(str(r.get('memory', r.get('content', ''))) for r in r2)
        overlap = len(r1_contents & r2_contents)
        print(f"    Page 1: {len(r1)} results, Page 2: {len(r2)} results, Overlap: {overlap}", flush=True)
        if overlap > 0:
            results['issues'].append(f"Pagination: {overlap} overlapping results")

    # ====================================================================
    # Chained graph connectivity check
    # ====================================================================
    print(f"\n  [Chain Connectivity]", flush=True)
    chain_entities = ['openmemory', 'qdrant', 'neo4j', 'ollama']
    for entity in chain_entities:
        try:
            resp = api_graph_context(entity)
            if resp.status_code == 200:
                data = resp.json()
                related = data.get('related_entities', {})
                total = sum(len(v) for v in related.values())
                root = data.get('root_entities', [])
                print(f"    '{entity}': {len(root)} root, {total} related", flush=True)
        except:
            print(f"    '{entity}': error", flush=True)

    # ====================================================================
    # Final summary
    # ====================================================================
    total_time = time.time() - test_start
    results['total_time'] = total_time
    results['total_memories'] = total_memories
    results['insert_time'] = insert_time

    # Compute total insert time (phases 1-13)
    phase_times = {k: v.get('time', 0) for k, v in results['phases'].items() if 'time' in v}
    total_insert = sum(v for k, v in phase_times.items() if k not in ('search',))

    print(f"\n  {'='*60}", flush=True)
    print(f"  SUMMARY for {model_name}:", flush=True)
    print(f"  {'='*60}", flush=True)
    print(f"  Total memories stored: {total_memories}", flush=True)
    print(f"  Total test time: {total_time:.1f}s", flush=True)
    print(f"  Insert time (phases 1-13): {total_insert:.1f}s", flush=True)
    print(f"  Avg search time: {avg_search:.0f}ms", flush=True)
    print(f"  Search: {search_pass}/{search_pass+search_fail} passed", flush=True)
    print(f"  Noise leaked: {results['phases'].get('noise', {}).get('leaked', '?')}", flush=True)
    print(f"  Dedup new: {results['phases'].get('dedup', {}).get('new', '?')}", flush=True)
    graph = results['phases'].get('graph', {})
    print(f"  Graph: {graph.get('nodes', '?')} nodes, {graph.get('edges', '?')} edges", flush=True)
    print(f"  Quality issues: {len(results.get('quality', []))}", flush=True)
    for q in results.get('quality', []):
        print(f"    - {q}", flush=True)
    print(f"  Issues ({len(results['issues'])}):", flush=True)
    for issue in results['issues']:
        print(f"    - {issue}", flush=True)

    # Cleanup for next model
    print(f"\n  Cleaning up...", flush=True)
    api_delete_all()
    time.sleep(3)
    unload_model(model_name)
    print(f"  Done.", flush=True)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Test only this model (substring match)')
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("COMPREHENSIVE SYSTEM BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    # Unload all first
    unload_all()

    models_to_test = MODELS
    if args.model:
        models_to_test = [(n, c) for n, c in MODELS if args.model in n]
        if not models_to_test:
            print(f"No model matching '{args.model}'")
            sys.exit(1)

    all_results = []
    for model_name, model_config in models_to_test:
        result = run_test(model_name, model_config)
        if result:
            all_results.append(result)

    # Final comparison
    if len(all_results) > 1:
        print(f"\n{'='*80}", flush=True)
        print("FINAL COMPARISON", flush=True)
        print(f"{'='*80}", flush=True)
        header = f"{'Model':<35} {'Mems':>5} {'Insert':>7} {'Search':>7} {'AvgQ':>6} {'Noise':>5} {'Dedup':>5} {'Graph':>12} {'Qual':>4} {'Iss':>3}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for r in sorted(all_results, key=lambda x: (len(x['issues']), len(x.get('quality', [])))):
            sp = r['phases'].get('search', {})
            search_str = f"{sp.get('pass',0)}/{sp.get('pass',0)+sp.get('fail',0)}"
            noise = r['phases'].get('noise', {}).get('leaked', '?')
            dedup = r['phases'].get('dedup', {}).get('new', '?')
            graph = r['phases'].get('graph', {})
            graph_str = f"{graph.get('nodes','?')}n/{graph.get('edges','?')}e"
            avg_ms = f"{sp.get('avg_ms', 0)}ms"
            insert_t = f"{r.get('insert_time', r['total_time']):.0f}s"
            qual = len(r.get('quality', []))
            iss = len(r['issues'])
            print(f"{r['model']:<35} {r['total_memories']:>5} {insert_t:>7} {search_str:>7} {avg_ms:>6} {noise:>5} {dedup:>5} {graph_str:>12} {qual:>4} {iss:>3}", flush=True)

        # Per-phase time comparison
        print(f"\n{'Phase Time Comparison (seconds)':^80}", flush=True)
        phase_names = ['single_facts', 'dense_chunk', 'multi_person', 'tech_stack',
                       'conversation', 'noise', 'negation_temporal', 'chained',
                       'short_facts', 'complex_tech']
        header2 = f"{'Phase':<20}" + "".join(f"{r['model'][:15]:>16}" for r in all_results)
        print(header2, flush=True)
        print("-" * len(header2), flush=True)
        for phase in phase_names:
            row = f"{phase:<20}"
            for r in all_results:
                t = r['phases'].get(phase, {}).get('time', 0)
                row += f"{t:>15.1f}s"
            print(row, flush=True)


if __name__ == '__main__':
    main()
