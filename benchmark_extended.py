"""
Extended deep testing benchmark — lots of insertions, quality inspection, speed tracking.

Usage: docker cp benchmark_extended.py openmemory-mcp:/usr/src/openmemory/ && \
       MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/benchmark_extended.py [--model MODEL]
"""
import requests
import json
import time
import sys
import os

API_URL = "http://localhost:8765/api"
OLLAMA_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
USER_ID = "steven"

MODELS = [
    'ministral-3:3b',
    'qwen3.5:4b',
    'qwen3.5:9b',
    'qwen3:4b-instruct-2507-q4_K_M',
]

# ============================================================================
# Extended test data — 30+ insertions covering all edge cases
# ============================================================================

TESTS = [
    # --- Category: Simple personal facts ---
    ("simple_personal", [
        "Steven is 28 years old.",
        "Steven has two monitors on his desk.",
        "Steven drinks black coffee, no sugar.",
        "Steven's birthday is on March 15th.",
        "Steven wears glasses.",
    ]),

    # --- Category: Preferences and opinions ---
    ("preferences", [
        "Steven strongly prefers Linux over Windows for development.",
        "Steven thinks Rust is the future of systems programming.",
        "Steven dislikes Python's GIL but loves its ecosystem.",
        "Steven's favorite game of all time is Dark Souls.",
        "Steven prefers mechanical keyboards with Cherry MX Brown switches.",
    ]),

    # --- Category: Professional background ---
    ("professional", [
        "Steven worked at Acme Corp from 2019 to 2024 as a senior PHP developer.",
        "Steven specializes in backend development with Laravel and Symfony.",
        "Steven has 7 years of professional programming experience.",
        "Steven earned a Bachelor's degree in Computer Science from Tallinn University of Technology.",
        "Steven freelances on the side building custom WordPress plugins.",
    ]),

    # --- Category: Echoes of the Fallen (dense project info) ---
    ("echoes_dense", [
        """Echoes of the Fallen is Steven's solo game project. It's a voxel-based roguelike
exploration game built with Godot Engine 4.3 using C++ for performance-critical systems
and GDScript for game logic. The game features procedural world generation using dual
contouring for terrain and wave function collapse for dungeon layouts.""",
        """The rendering pipeline in Echoes uses a deferred shading approach with SSAO,
volumetric fog, and temporal anti-aliasing. The voxel engine uses an octree structure
for LOD management. Chunk meshing runs on 8 threads via WorkerThreadPool, achieving
sub-16ms frame times with 256x256x64 chunks.""",
        """The game's narrative system uses a knowledge graph where players permanently
unlock lore entries that persist across death. Items and equipment are temporary —
lost on death. This creates a roguelike loop where player knowledge is the real
progression, not gear.""",
    ]),

    # --- Category: Multi-person complex relationships ---
    ("multi_person", [
        "Jake is the lead DevOps engineer at CloudTech Inc and Steven's closest collaborator.",
        "Maria is a UX designer who previously worked with Steven at Acme Corp.",
        "Tom is Jake's manager at CloudTech and has approved the open-source budget.",
        "Steven and Jake co-maintain the OpenMemory project on GitHub.",
        "Maria designed the UI for OpenMemory's dashboard.",
    ]),

    # --- Category: Technical stack (impersonal) ---
    ("tech_impersonal", [
        "The API server uses FastAPI with Pydantic v2 for request validation.",
        "PostgreSQL 16 handles persistent storage with pgvector extension for embeddings.",
        "Redis 7.2 manages rate limiting, caching, and WebSocket pub/sub.",
        "The deployment uses Docker Compose with 6 services and Traefik as reverse proxy.",
        "CI/CD runs on GitHub Actions: lint, typecheck, test, build, deploy to staging.",
    ]),

    # --- Category: Negations and temporal changes ---
    ("negations_temporal", [
        "Steven stopped using Windows in 2023 and switched to Arch Linux.",
        "Steven rejected React after trying it for 3 months in favor of Svelte.",
        "Steven used to smoke but quit in 2021.",
        "Steven previously drove a Toyota but now cycles to work.",
        "Steven no longer uses MongoDB — switched to PostgreSQL for everything.",
    ]),

    # --- Category: Very short facts ---
    ("very_short", [
        "Steven has a cat named Luna.",
        "Steven speaks Estonian.",
        "Steven likes hiking.",
        "Steven plays chess.",
        "Steven uses Neovim.",
    ]),

    # --- Category: Conversation pairs ---
    ("conversations", [
        ("I've been experimenting with Zig lately. The comptime feature is incredible — it replaces C macros entirely.",
         "Zig's comptime is quite powerful. Many C developers are excited about it as a modern alternative. Have you tried it for any real projects?"),
        ("Not yet, just toy programs. I want to rewrite the voxel mesher in Zig eventually, but Godot's GDExtension only supports C++ right now.",
         "That makes sense. You'd need to write a C ABI wrapper. Some people have done Zig GDExtension bindings already."),
        ("I also started learning Blender for creating 3D assets. The sculpting tools are amazing but the UV unwrapping is painful.",
         "UV unwrapping is notoriously tedious. Have you looked at automatic UV tools like UVPackmaster or Zen UV?"),
    ]),

    # --- Category: Noise that should NOT create memories ---
    ("noise", [
        "Hello!",
        "Thanks for your help.",
        "Can you repeat that?",
        "OK, got it.",
        "What's the weather like?",
        "Sure, let me think about it.",
        "Hmm, interesting.",
    ]),

    # --- Category: Contradictions (should UPDATE existing facts) ---
    ("contradictions", [
        "Steven is now 29 years old.",  # Was 28
        "Steven moved from Tallinn to Tartu last month.",  # Was Tallinn
        "Steven switched from Arch Linux to NixOS.",  # Was Arch
        "Steven sold his mechanical keyboard and now uses a laptop keyboard.",  # Was MX Brown
    ]),

    # --- Category: Edge cases ---
    ("edge_cases", [
        # Mixed languages
        "Steven elab Tartus ja töötab kodus. He codes mostly in PHP and Rust.",
        # Very long single fact
        "Steven's home office setup consists of a custom-built PC with an AMD Ryzen 9 7950X processor, 64GB DDR5 RAM, an NVIDIA RTX 4090 GPU, two 32-inch 4K monitors from LG, a Keychron Q1 Pro keyboard, a Logitech MX Master 3 mouse, and a Herman Miller Aeron chair.",
        # Technical jargon heavy
        "The ECS architecture in Echoes uses a sparse set storage backend with archetype-based iteration, achieving O(1) component access and cache-friendly iteration over 100k entities at 144fps.",
        # Emotional/subjective
        "Steven finds debugging memory leaks in C++ extremely frustrating but rewarding when fixed.",
        # Future plans
        "Steven plans to attend GDC 2027 in San Francisco to showcase Echoes of the Fallen.",
    ]),

    # --- Category: Dedup stress test (variations of known facts) ---
    ("dedup_stress", [
        "Steven is a PHP developer.",  # Exact duplicate
        "Steven works as a PHP developer.",  # Slight variation
        "Steven programs in PHP professionally.",  # Paraphrase
        "Steven has a cat called Luna.",  # Was "named Luna"
        "Steven uses the Neovim text editor.",  # Was just "Neovim"
    ]),
]


# ============================================================================
# Helpers
# ============================================================================

def switch_model(model_name):
    resp = requests.put(f"{API_URL}/v1/config/mem0/llm", json={
        "provider": "ollama",
        "config": {"model": model_name, "temperature": 0, "max_tokens": 4000, "ollama_base_url": OLLAMA_URL}
    }, timeout=10)
    if resp.status_code == 200:
        r2 = requests.get(f"{API_URL}/v1/config/mem0/llm", timeout=5)
        return r2.json().get('config', {}).get('model') == model_name
    return False

def delete_all():
    requests.post(f"{API_URL}/v1/memories/delete_all", json={"user_id": USER_ID}, timeout=120)
    time.sleep(3)

def add_memory(text, timeout=60):
    t0 = time.time()
    resp = requests.post(f"{API_URL}/v1/memories/", json={
        "text": text, "user_id": USER_ID, "app_id": "openmemory", "infer": True,
    }, timeout=timeout)
    return time.time() - t0, resp

def add_conversation(user_msg, llm_resp, timeout=60):
    t0 = time.time()
    resp = requests.post(f"{API_URL}/v1/memories/conversation", json={
        "user_message": user_msg, "llm_response": llm_resp, "user_id": USER_ID,
    }, timeout=timeout)
    return time.time() - t0, resp

def count_memories():
    try:
        r = requests.get(f"{API_URL}/v1/memories/", params={"user_id": USER_ID, "size": 1, "page": 1}, timeout=10)
        return r.json().get('total', 0) if r.status_code == 200 else 0
    except:
        return 0

def get_all_memories():
    """Get all memory texts, paginating if needed (max page size is 100)."""
    all_items = []
    page = 1
    while True:
        try:
            r = requests.get(f"{API_URL}/v1/memories/", params={
                "user_id": USER_ID, "size": 100, "page": page,
            }, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            items = data.get('items', [])
            all_items.extend(item.get('content', '') for item in items)
            if page >= data.get('pages', 1):
                break
            page += 1
        except:
            break
    return all_items

def search(query):
    t0 = time.time()
    r = requests.post(f"{API_URL}/v1/memories/search", json={
        "query": query, "user_id": USER_ID, "app_id": "openmemory", "limit": 10,
    }, timeout=30)
    dt = time.time() - t0
    results = r.json().get('results', []) if r.status_code == 200 else []
    return dt, results

def graph_stats():
    try:
        r = requests.get(f"{API_URL}/v1/memories/graph/stats", params={"user_id": USER_ID}, timeout=30)
        if r.status_code == 200:
            s = r.json()
            return s.get('total_entities', 0), s.get('total_relationships', 0), s.get('entity_types', [])
    except:
        pass
    return 0, 0, []

def unload_model(model):
    try:
        requests.post(f'{OLLAMA_URL}/api/chat', json={
            'model': model, 'messages': [{'role': 'user', 'content': 'x'}],
            'stream': False, 'think': False, 'options': {'num_predict': 1}, 'keep_alive': 0,
        }, timeout=30)
    except:
        pass


# ============================================================================
# Main test runner
# ============================================================================

def run_model_test(model_name):
    print(f"\n{'='*80}", flush=True)
    print(f"  EXTENDED TEST: {model_name}", flush=True)
    print(f"{'='*80}", flush=True)

    if not switch_model(model_name):
        print(f"  FAILED to switch model — skipping", flush=True)
        return None

    delete_all()
    assert count_memories() == 0, "Cleanup failed"

    results = {
        'model': model_name,
        'categories': {},
        'timings': [],
        'total_memories': 0,
        'quality_issues': [],
    }
    total_start = time.time()

    # ---- Insert all test data ----
    for cat_name, items in TESTS:
        cat_start = time.time()
        before = count_memories()
        cat_times = []

        print(f"\n  [{cat_name}] ({len(items)} inputs)", flush=True)

        for item in items:
            if isinstance(item, tuple):
                # Conversation pair
                dt, resp = add_conversation(item[0], item[1], timeout=120)
                cat_times.append(dt)
                if resp.status_code == 200:
                    d = resp.json()
                    stored = d.get('facts_stored', 0)
                    deduped = d.get('duplicates_skipped', 0)
                    print(f"    {dt:.1f}s  conv: +{stored} stored, {deduped} deduped", flush=True)
                else:
                    print(f"    {dt:.1f}s  conv: HTTP {resp.status_code}", flush=True)
            else:
                # Regular text
                dt, resp = add_memory(item, timeout=120)
                cat_times.append(dt)
                short = item[:65].replace('\n', ' ')
                if resp.status_code == 200:
                    print(f"    {dt:.1f}s  \"{short}\"", flush=True)
                else:
                    print(f"    {dt:.1f}s  ERR:{resp.status_code} \"{short}\"", flush=True)

        # Wait for async graph
        time.sleep(4)
        after = count_memories()
        cat_new = after - before
        cat_time = time.time() - cat_start
        avg_time = sum(cat_times) / len(cat_times) if cat_times else 0

        print(f"    => +{cat_new} memories, {cat_time:.1f}s total, {avg_time:.2f}s avg/insert", flush=True)

        results['categories'][cat_name] = {
            'inputs': len(items),
            'new_memories': cat_new,
            'total_time': round(cat_time, 1),
            'avg_time': round(avg_time, 2),
            'times': [round(t, 2) for t in cat_times],
        }
        results['timings'].extend(cat_times)

    total_memories = count_memories()
    insert_time = time.time() - total_start
    results['total_memories'] = total_memories
    results['insert_time'] = round(insert_time, 1)

    # ---- Quality inspection ----
    print(f"\n  [QUALITY INSPECTION] {total_memories} memories", flush=True)
    all_mems = get_all_memories()

    # Check first-person leaks
    fp_leaks = []
    for m in all_mems:
        words = m.split()
        if any(w in ('I', 'My', 'my', 'me', 'Me') for w in words):
            if 'steven' not in m.lower() and 'user' not in m.lower():
                fp_leaks.append(m)
    if fp_leaks:
        print(f"    ISSUE: {len(fp_leaks)} first-person leaks (I/My/me):", flush=True)
        for m in fp_leaks[:5]:
            print(f"      - {m[:90]}", flush=True)
        results['quality_issues'].append(f"{len(fp_leaks)} first-person leaks")

    # Check "User" references
    user_refs = [m for m in all_mems if m.lower().startswith('user ') or 'the user' in m.lower()]
    if user_refs:
        print(f"    ISSUE: {len(user_refs)} 'User' references:", flush=True)
        for m in user_refs[:3]:
            print(f"      - {m[:90]}", flush=True)
        results['quality_issues'].append(f"{len(user_refs)} 'User' references")

    # Check orphaned facts (no subject)
    orphaned = []
    for m in all_mems:
        first = m.split()[0].lower() if m.split() else ""
        if first in ('uses', 'has', 'is', 'was', 'does', 'the', 'a', 'total', 'it', 'runs', 'built', 'features'):
            orphaned.append(m)
    if orphaned:
        print(f"    ISSUE: {len(orphaned)} context-stripped facts:", flush=True)
        for m in orphaned[:5]:
            print(f"      - {m[:90]}", flush=True)
        results['quality_issues'].append(f"{len(orphaned)} context-stripped facts")

    # Check for hallucinated content (facts about things NOT in test data)
    hallucination_keywords = ['python developer', 'java developer', 'works at google', 'lives in london']
    hallucinated = []
    for m in all_mems:
        ml = m.lower()
        for kw in hallucination_keywords:
            if kw in ml:
                hallucinated.append(m)
                break
    if hallucinated:
        print(f"    ISSUE: {len(hallucinated)} possible hallucinations:", flush=True)
        for m in hallucinated[:3]:
            print(f"      - {m[:90]}", flush=True)
        results['quality_issues'].append(f"{len(hallucinated)} hallucinations")

    if not fp_leaks and not user_refs and not orphaned and not hallucinated:
        print(f"    All {total_memories} memories passed quality checks", flush=True)

    # ---- Dump all memories for manual review ----
    print(f"\n  [MEMORY DUMP] All {total_memories} stored memories:", flush=True)
    for i, m in enumerate(sorted(all_mems), 1):
        print(f"    {i:3d}. {m}", flush=True)

    # ---- Search quality ----
    print(f"\n  [SEARCH QUALITY]", flush=True)
    search_queries = [
        ("Steven's age", ["28", "29"]),
        ("cat pet", ["luna"]),
        ("game engine Godot", ["godot", "echoes"]),
        ("PHP work history", ["php", "acme"]),
        ("keyboard preference", ["keyboard", "keychron", "cherry"]),
        ("stopped using", ["windows", "mongodb", "react"]),
        ("Jake collaborator", ["jake", "cloudtech", "devops"]),
        ("Maria designer", ["maria", "ui", "ux"]),
        ("Zig programming", ["zig", "comptime"]),
        ("Blender 3D", ["blender", "sculpt"]),
        ("NixOS Linux", ["nixos", "arch", "linux"]),
        ("home office setup", ["ryzen", "4090", "aeron", "keychron"]),
        ("GDC plans", ["gdc", "2027", "san francisco"]),
        ("voxel ECS", ["ecs", "archetype", "sparse"]),
        ("Tartu Estonia", ["tartu"]),
    ]
    search_pass = 0
    search_fail = 0
    search_times = []
    for query, expected in search_queries:
        dt, res = search(query)
        search_times.append(dt)
        all_text = ' '.join(str(r) for r in res).lower()
        found = any(kw.lower() in all_text for kw in expected)
        status = "PASS" if found else "MISS"
        if found:
            search_pass += 1
        else:
            search_fail += 1
        top_result = str(res[0].get('memory', res[0].get('content', '')))[:60] if res else "none"
        print(f"    [{status}] '{query}' -> {len(res)} results, {dt*1000:.0f}ms  top: \"{top_result}\"", flush=True)

    results['search_pass'] = search_pass
    results['search_fail'] = search_fail
    results['avg_search_ms'] = round(sum(search_times) / len(search_times) * 1000) if search_times else 0

    # ---- Graph stats ----
    print(f"\n  [GRAPH]", flush=True)
    nodes, edges, types = graph_stats()
    print(f"    Nodes: {nodes}, Edges: {edges}", flush=True)
    if types:
        type_str = ', '.join(f"{t['type']}:{t['count']}" for t in types[:12])
        print(f"    Types: {type_str}", flush=True)
    results['graph_nodes'] = nodes
    results['graph_edges'] = edges

    # Self-ref check
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get('NEO4J_URL', 'bolt://neo4j:7687'),
            auth=(os.environ.get('NEO4J_USER', 'neo4j'), os.environ.get('NEO4J_PASSWORD', 'openmemory'))
        )
        with driver.session() as sess:
            sr = sess.run("MATCH (n)-[r]->(n) WHERE n.user_id = $u RETURN count(r) as c", u=USER_ID).single()
            self_refs = sr['c'] if sr else 0
            print(f"    Self-ref edges: {self_refs}", flush=True)
            results['self_refs'] = self_refs
        driver.close()
    except Exception as e:
        print(f"    Self-ref check: {e}", flush=True)

    # ---- Speed summary ----
    all_times = results['timings']
    print(f"\n  [SPEED SUMMARY]", flush=True)
    print(f"    Total inserts: {len(all_times)}", flush=True)
    print(f"    Total insert time: {insert_time:.1f}s", flush=True)
    print(f"    Avg per insert: {sum(all_times)/len(all_times):.2f}s", flush=True)
    print(f"    Min: {min(all_times):.2f}s, Max: {max(all_times):.2f}s", flush=True)
    print(f"    Median: {sorted(all_times)[len(all_times)//2]:.2f}s", flush=True)
    print(f"    Avg search: {results['avg_search_ms']}ms", flush=True)

    # Breakdown by category
    print(f"\n    Per-category avg insert time:", flush=True)
    for cat_name, cat_data in results['categories'].items():
        print(f"      {cat_name:<20} {cat_data['avg_time']:.2f}s  (+{cat_data['new_memories']} mems from {cat_data['inputs']} inputs)", flush=True)

    # ---- Final summary ----
    total_time = time.time() - total_start
    results['total_time'] = round(total_time, 1)

    print(f"\n  {'='*60}", flush=True)
    print(f"  FINAL: {model_name}", flush=True)
    print(f"  {'='*60}", flush=True)
    print(f"  Memories: {total_memories}", flush=True)
    print(f"  Quality issues: {len(results['quality_issues'])}", flush=True)
    for q in results['quality_issues']:
        print(f"    - {q}", flush=True)
    print(f"  Search: {search_pass}/{search_pass+search_fail}", flush=True)
    print(f"  Graph: {nodes}n/{edges}e, {self_refs if 'self_refs' in dir() else '?'} self-refs", flush=True)
    print(f"  Speed: {sum(all_times)/len(all_times):.2f}s avg insert, {results['avg_search_ms']}ms avg search", flush=True)
    print(f"  Total time: {total_time:.1f}s", flush=True)

    # Cleanup
    print(f"\n  Cleaning up...", flush=True)
    delete_all()
    unload_model(model_name)
    print(f"  Done.\n", flush=True)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Test only this model (substring match)')
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("  EXTENDED DEEP BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    # Unload all
    for m in MODELS:
        unload_model(m)

    models = MODELS
    if args.model:
        models = [m for m in MODELS if args.model in m]
        if not models:
            print(f"No model matching '{args.model}'")
            sys.exit(1)

    all_results = []
    for model in models:
        result = run_model_test(model)
        if result:
            all_results.append(result)

    # ========================================================================
    # Cross-model comparison
    # ========================================================================
    if len(all_results) > 1:
        print(f"\n{'='*80}", flush=True)
        print(f"  CROSS-MODEL COMPARISON", flush=True)
        print(f"{'='*80}", flush=True)

        # Main table
        print(f"\n{'Model':<35} {'Mems':>5} {'Qual':>4} {'Srch':>6} {'Graph':>10} {'SRef':>4} {'AvgIns':>7} {'AvgSrch':>7} {'Total':>6}", flush=True)
        print("-" * 90, flush=True)
        for r in all_results:
            srch = f"{r['search_pass']}/{r['search_pass']+r['search_fail']}"
            graph = f"{r['graph_nodes']}n/{r['graph_edges']}e"
            avg_ins = f"{sum(r['timings'])/len(r['timings']):.2f}s"
            avg_srch = f"{r['avg_search_ms']}ms"
            sr = r.get('self_refs', '?')
            print(f"{r['model']:<35} {r['total_memories']:>5} {len(r['quality_issues']):>4} {srch:>6} {graph:>10} {sr:>4} {avg_ins:>7} {avg_srch:>7} {r['total_time']:>5.0f}s", flush=True)

        # Per-category memory count comparison
        print(f"\n{'Memories per category:'}", flush=True)
        cat_names = list(all_results[0]['categories'].keys())
        header = f"  {'Category':<20}" + "".join(f"{r['model'][:14]:>15}" for r in all_results)
        print(header, flush=True)
        print("  " + "-" * (20 + 15 * len(all_results)), flush=True)
        for cat in cat_names:
            row = f"  {cat:<20}"
            for r in all_results:
                c = r['categories'].get(cat, {})
                row += f"{c.get('new_memories', 0):>15}"
            print(row, flush=True)

        # Per-category speed comparison
        print(f"\n{'Avg insert time per category (seconds):'}", flush=True)
        header = f"  {'Category':<20}" + "".join(f"{r['model'][:14]:>15}" for r in all_results)
        print(header, flush=True)
        print("  " + "-" * (20 + 15 * len(all_results)), flush=True)
        for cat in cat_names:
            row = f"  {cat:<20}"
            for r in all_results:
                c = r['categories'].get(cat, {})
                row += f"{c.get('avg_time', 0):>14.2f}s"
            print(row, flush=True)

        # Verdict
        print(f"\n  VERDICT:", flush=True)
        best_mems = max(all_results, key=lambda r: r['total_memories'])
        best_speed = min(all_results, key=lambda r: sum(r['timings'])/len(r['timings']))
        best_graph = max(all_results, key=lambda r: r['graph_edges'])
        best_quality = min(all_results, key=lambda r: len(r['quality_issues']))
        best_search = max(all_results, key=lambda r: r['search_pass'])

        print(f"    Most memories:  {best_mems['model']} ({best_mems['total_memories']})", flush=True)
        print(f"    Fastest insert: {best_speed['model']} ({sum(best_speed['timings'])/len(best_speed['timings']):.2f}s avg)", flush=True)
        print(f"    Richest graph:  {best_graph['model']} ({best_graph['graph_edges']} edges)", flush=True)
        print(f"    Best search:    {best_search['model']} ({best_search['search_pass']}/{best_search['search_pass']+best_search['search_fail']})", flush=True)
        print(f"    Best quality:   {best_quality['model']} ({len(best_quality['quality_issues'])} issues)", flush=True)


if __name__ == '__main__':
    main()
