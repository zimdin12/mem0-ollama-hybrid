#!/usr/bin/env python3
"""Deep quality comparison of extraction models — with model-specific params."""
import requests, json, time

OLLAMA = 'http://ollama:11434'

MODELS = [
    'qwen3:4b-instruct-2507-q4_K_M',
    'qwen3.5:4b',
    'qwen35-9b-highiq',
]


def detect_family(model):
    name = model.lower()
    if 'qwen3.5' in name or 'qwen35' in name:
        return 'qwen3.5'
    if 'qwen3' in name:
        return 'qwen3'
    return 'unknown'


def get_options(model):
    if detect_family(model) == 'qwen3.5':
        return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}
    return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}

TESTS = {
    # ---- EASY (should all handle) ----
    "basic_facts": {
        "input": "Steven lives in Tallinn, Estonia. He has a cat named Pixel. His birthday is March 15th.",
        "expect": "3 clean facts with Steven as subject",
    },

    # ---- MEDIUM (pronoun resolution, negation) ----
    "negation": {
        "input": "Steven does NOT use Visual Studio Code anymore. He switched from VS Code to Neovim last month. He tried Emacs but didn't like it. He has never used IntelliJ.",
        "expect": "4 negation-preserving facts. Graph: stopped_using VS Code, uses Neovim, tried Emacs, never_used IntelliJ",
    },

    # ---- HARD (complex multi-hop reasoning) ----
    "complex_project": {
        "input": """Echoes of the Fallen is a voxel roguelike built with Unreal Engine 5 and C++.
The rendering pipeline uses Voxel Plugin 2.0 which leverages UE5's Nanite for virtualized geometry and Lumen for global illumination.
Mesh generation uses dual contouring instead of marching cubes because it preserves sharp edges for architectural ruins.
The chunk system is 32x32x32 with octree LOD, greedy meshing for 40% triangle reduction, and async generation on background threads.
Steven chose this tech stack after evaluating Unity with Cubiquity and Godot with VoxelTools, rejecting both due to limited LOD support.""",
        "expect": "Should capture: project->uses->VoxelPlugin->for->UE5, dual contouring OVER marching cubes, rejected Unity+Cubiquity and Godot+VoxelTools",
    },

    "interconnected_people": {
        "input": """Steven, Maria, and Jake are building a game together. Steven handles all C++ gameplay programming and Unreal blueprints.
Maria is the art director who creates voxel assets in MagicaVoxel and does character animation in Blender.
Jake manages the CI/CD pipeline using GitHub Actions and Docker, and also handles QA testing on both Windows and Linux.
Maria reports to Steven as project lead but has full creative control over the art direction.
Jake previously worked at Ubisoft Montreal for 3 years before joining as a freelancer.
Steven and Maria met at the Nordic Game Conference in 2024 where Maria was presenting her talk on procedural voxel art.""",
        "expect": "6 people-tool relationships, reporting structure, Jake's history, conference meeting. No cross-contamination.",
    },

    "subtle_corrections": {
        "input": """The game was originally called 'Fallen Echoes' but Steven renamed it to 'Echoes of the Fallen' because the original name was too similar to an existing mobile game.
Development started in January 2025 with a 12-month timeline, but after adding multiplayer architecture the scope expanded to 18 months with a target release of July 2026.
The initial budget was $500 for asset packs but grew to $1,500 after purchasing Voxel Plugin Pro ($200), Metasounds audio pack ($80), and hiring Maria for contract art ($720).
Steven first planned Steam-only release but added Epic Games Store after learning about their 12% revenue split versus Steam's 30%.""",
        "expect": "Current values (not old): Echoes of the Fallen, 18 months, July 2026, $1500, Steam+Epic. Should capture WHY for each change.",
    },

    "technical_architecture": {
        "input": """The memory system architecture uses three storage layers: Qdrant for vector similarity search with 1024-dimensional cosine embeddings,
Neo4j for entity-relationship graph traversal using Cypher queries, and SQLite for temporal ordering and metadata.
The extraction pipeline has five stages: regex fact splitting, LLM fact review with qwen3:4b, deduplication against existing vectors at 0.85 cosine threshold,
graph entity extraction via JSON mode, and async background indexing.
Search combines results from all three stores using a 60/30/10 weighting for vector/graph/temporal sources with score normalization and interleaving.
The Ollama container runs on an RTX 4090 with 24GB VRAM using flash attention and q8_0 KV cache quantization for 2x context window at minimal quality loss.""",
        "expect": "Must preserve: 3 storage layers with specific tech, 5 pipeline stages, 60/30/10 weighting, 0.85 threshold, RTX 4090 specs",
    },

    "emotional_with_decisions": {
        "input": """I spent 3 days debugging why Ollama's tool calling fails with qwen3 models. Incredibly frustrating.
The root cause is that qwen3:4b returns tool call syntax as plain text instead of structured tool_calls objects.
I ended up replacing the entire tool-calling pipeline with JSON mode extraction, which works perfectly.
Honestly the JSON approach is better anyway - single LLM call instead of 3-4 sequential tool calls, and the output is more consistent.
Considering switching the embedding model from nomic-embed-text to qwen3-embedding:0.6b for better multilingual support.
Already tested it - same quality for English facts, much better for Estonian and Russian text, and 5.7x faster.""",
        "expect": "Extract decisions and reasoning: tool calling broken, replaced with JSON mode (why: single call, consistent), considering embedding switch (why: multilingual, tested), 5.7x speed diff",
    },

    "ambiguous_references": {
        "input": """Steven's game uses a skill system inspired by isekai anime where death triggers skill evolution.
During gameplay, actions like swimming generate water affinity skills. Repeated falling builds gliding.
When the player dies, fresh skills from that run become permanent - they call this the 'ripening' mechanic.
The system creates interesting emergent gameplay: a player who falls into water a lot might develop both swimming AND water breathing,
which combines into deep diving - a hybrid skill that unlocks underwater cave exploration.
Steven says it's the core differentiator from competitors like Hades and Dead Cells which use item-based progression.""",
        "expect": "Must connect skills to the game by name. Capture: ripening mechanic, skill combination (water+falling=diving), competitive positioning vs Hades/Dead Cells",
    },

    "mixed_languages": {
        "input": """Steven codes primarily in English but writes commit messages mixing Estonian and English.
His variable naming convention is camelCase for C++ (following UE5 style) and snake_case for Python and Rust.
He uses 'tere' (Estonian for hello) as his test greeting in all language-related unit tests.
His Neo4j database has entity names in English but some user-facing labels in Estonian like 'mänguprojekt' (game project).
The README files are in English but he maintains an Estonian FAQ at docs/KKK.md (KKK = Korduma Kippuvad Küsimused).""",
        "expect": "Code conventions per language, Estonian terms with translations, file paths",
    },
}

FACT_SYSTEM = """Extract all factual information from the text. Return each fact on a new line.
Rules:
1. Each fact must be a complete, standalone sentence with the subject named.
2. Preserve negations (does NOT, never, stopped, rejected).
3. Preserve reasons and conditions (because, due to, after, instead of).
4. Preserve numbers, measurements, percentages, and thresholds exactly.
5. Do NOT make up information not in the text.
6. Do NOT add explanations or commentary.
7. Replace "I/me/my" with the person's name if identifiable from context."""

GRAPH_SYSTEM = """Extract entities and relationships from text. Return ONLY a JSON object.
Format: {"entities": [{"name": "...", "type": "..."}, ...], "relationships": [{"source": "...", "relation": "...", "target": "..."}, ...]}
Entity types: person, project, technology, tool, framework, language, hardware, organization, place, game_element, algorithm, database, service
Only person, project, and game_element entities should be relationship sources."""

# Extra instruction appended for qwen3.5 family (tends to return empty relationships)
GRAPH_QWEN35_SUFFIX = """

IMPORTANT: You MUST generate relationships for every entity. Every entity should connect to at least one person or project via a relationship. If an entity has no relationship, remove it from entities too."""


def strip_fences(text):
    text = text.strip()
    if text.startswith('```'):
        nl = text.find('\n')
        if nl != -1: text = text[nl + 1:]
        if text.rstrip().endswith('```'): text = text.rstrip()[:-3].rstrip()
    return text


def test_model(model):
    family = detect_family(model)
    opts = get_options(model)
    graph_sys = GRAPH_SYSTEM + (GRAPH_QWEN35_SUFFIX if family == 'qwen3.5' else '')

    # Unload all
    ps = requests.get(f'{OLLAMA}/api/ps').json()
    for m in ps.get('models', []):
        requests.post(f'{OLLAMA}/api/generate', json={'model': m['name'], 'keep_alive': 0})
    time.sleep(3)

    # Warm up
    requests.post(f'{OLLAMA}/api/chat', json={
        'model': model, 'messages': [{'role': 'user', 'content': 'Ready'}],
        'stream': False, 'think': False,
    }, timeout=120)

    results = {}
    for test_name, test in TESTS.items():
        text = test['input']

        # Fact extraction
        start = time.time()
        r = requests.post(f'{OLLAMA}/api/chat', json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': FACT_SYSTEM},
                {'role': 'user', 'content': f'Extract facts from:\n{text}'},
            ],
            'stream': False, 'think': False,
            'options': opts,
        }, timeout=120)
        fact_time = time.time() - start
        facts_raw = r.json()['message']['content']

        # Graph extraction
        start = time.time()
        r = requests.post(f'{OLLAMA}/api/chat', json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': graph_sys},
                {'role': 'user', 'content': f'Extract from:\n{text}'},
            ],
            'stream': False, 'format': 'json', 'think': False,
            'options': opts,
        }, timeout=120)
        graph_time = time.time() - start
        graph_raw = r.json()['message']['content']

        try:
            graph = json.loads(strip_fences(graph_raw))
            ents = graph.get('entities', [])
            rels = graph.get('relationships', [])
        except:
            ents, rels = [], []

        facts_lines = [l.strip() for l in facts_raw.split('\n')
                       if l.strip() and len(l.strip()) > 15
                       and not l.strip().startswith('#')
                       and not l.strip().startswith('|')
                       and '**:' not in l.strip()
                       and 'extracted' not in l.strip().lower()[:20]]

        results[test_name] = {
            'fact_time': fact_time, 'graph_time': graph_time,
            'fact_count': len(facts_lines), 'facts': facts_lines,
            'facts_raw': facts_raw,
            'ent_count': len(ents), 'rel_count': len(rels),
            'entities': ents, 'relationships': rels,
            'graph_raw': graph_raw,
        }
    return results


def print_comparison(test_name, all_results):
    test = TESTS[test_name]
    print(f'\n{"="*80}')
    print(f'TEST: {test_name}')
    print(f'{"="*80}')
    # Show first 150 chars of input
    inp = test["input"].replace('\n', ' ')[:150]
    print(f'INPUT: {inp}...')
    print(f'EXPECT: {test["expect"]}')

    for model in MODELS:
        short = model.split(':')[0]
        r = all_results[model][test_name]
        print(f'\n  [{short}] {r["fact_count"]}f/{r["ent_count"]}e/{r["rel_count"]}r '
              f'(fact:{r["fact_time"]:.1f}s graph:{r["graph_time"]:.1f}s)')
        for f in r['facts'][:10]:
            f = f.lstrip('0123456789.-) *')
            print(f'    F: {f[:100]}')
        for e in r['entities'][:8]:
            print(f'    E: {e.get("name","?")} ({e.get("type","?")})')
        for rel in r['relationships'][:8]:
            print(f'    R: {rel.get("source","?")} --{rel.get("relation","?")}--> {rel.get("target","?")}')
        if r['ent_count'] == 0 and r['rel_count'] == 0:
            print(f'    GRAPH EMPTY/ERROR')


if __name__ == '__main__':
    all_results = {}
    for model in MODELS:
        print(f'\nTesting: {model}...')
        all_results[model] = test_model(model)
        avg_f = sum(r['fact_time'] for r in all_results[model].values()) / len(TESTS)
        avg_g = sum(r['graph_time'] for r in all_results[model].values()) / len(TESTS)
        print(f'  Done. avg fact={avg_f:.2f}s graph={avg_g:.2f}s')

    for test_name in TESTS:
        print_comparison(test_name, all_results)

    # Summary
    print(f'\n\n{"="*80}')
    print(f'SUMMARY')
    print(f'{"="*80}')
    header = f'{"Test":<25}'
    for m in MODELS:
        header += f'  {m.split(":")[0][-16:]:>18}'
    print(header)
    print('-' * len(header))

    totals = {m: {'f': 0, 'e': 0, 'r': 0, 'ft': 0, 'gt': 0} for m in MODELS}
    for test_name in TESTS:
        row = f'{test_name:<25}'
        for model in MODELS:
            r = all_results[model][test_name]
            row += f'  {r["fact_count"]:>2}f/{r["ent_count"]:>2}e/{r["rel_count"]:>2}r '
            row += f'{r["fact_time"]+r["graph_time"]:>4.1f}s'
            totals[model]['f'] += r['fact_count']
            totals[model]['e'] += r['ent_count']
            totals[model]['r'] += r['rel_count']
            totals[model]['ft'] += r['fact_time']
            totals[model]['gt'] += r['graph_time']
        print(row)

    print('-' * len(header))
    row = f'{"TOTAL":<25}'
    for model in MODELS:
        t = totals[model]
        row += f'  {t["f"]:>2}f/{t["e"]:>2}e/{t["r"]:>2}r '
        row += f'{t["ft"]+t["gt"]:>4.1f}s'
    print(row)
