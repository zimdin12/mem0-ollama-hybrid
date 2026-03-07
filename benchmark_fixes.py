#!/usr/bin/env python3
"""Quick benchmark: before/after fixes for qwen3 and qwen3.5 graph extraction."""
import requests, json, time

OLLAMA = 'http://ollama:11434'

def detect_family(model):
    name = model.lower()
    if 'qwen3.5' in name or 'qwen35' in name: return 'qwen3.5'
    if 'qwen3' in name: return 'qwen3'
    return 'unknown'

def get_options(model):
    if detect_family(model) == 'qwen3.5':
        return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}
    return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}

GRAPH_SYS = """Extract entities and relationships from text. Return ONLY a JSON object.
Format: {"entities": [{"name": "...", "type": "..."}, ...], "relationships": [{"source": "...", "relation": "...", "target": "..."}, ...]}
Entity types: person, project, technology, tool, framework, language, hardware, organization, place, game_element, algorithm, database, service
Only person, project, and game_element entities should be relationship sources."""

GRAPH_35_SUFFIX = """

IMPORTANT: You MUST generate relationships for every entity. Every entity should connect to at least one person or project via a relationship. If an entity has no relationship, remove it from entities too."""

TESTS = {
    "basic_facts": "Steven lives in Tallinn, Estonia. He has a cat named Pixel. His birthday is March 15th.",
    "negation": "Steven does NOT use Visual Studio Code anymore. He switched from VS Code to Neovim last month. He tried Emacs but didn't like it. He has never used IntelliJ.",
    "complex_project": """Echoes of the Fallen is a voxel roguelike built with Unreal Engine 5 and C++.
The rendering pipeline uses Voxel Plugin 2.0 which leverages UE5's Nanite for virtualized geometry and Lumen for global illumination.
Mesh generation uses dual contouring instead of marching cubes because it preserves sharp edges for architectural ruins.
The chunk system is 32x32x32 with octree LOD, greedy meshing for 40% triangle reduction, and async generation on background threads.
Steven chose this tech stack after evaluating Unity with Cubiquity and Godot with VoxelTools, rejecting both due to limited LOD support.""",
    "subtle_corrections": """The game was originally called 'Fallen Echoes' but Steven renamed it to 'Echoes of the Fallen' because the original name was too similar to an existing mobile game.
Development started in January 2025 with a 12-month timeline, but after adding multiplayer architecture the scope expanded to 18 months with a target release of July 2026.
The initial budget was $500 for asset packs but grew to $1,500 after purchasing Voxel Plugin Pro ($200), Metasounds audio pack ($80), and hiring Maria for contract art ($720).
Steven first planned Steam-only release but added Epic Games Store after learning about their 12% revenue split versus Steam's 30%.""",
}


def strip_fences(text):
    text = text.strip()
    if text.startswith('```'):
        nl = text.find('\n')
        if nl != -1: text = text[nl + 1:]
        if text.rstrip().endswith('```'): text = text.rstrip()[:-3].rstrip()
    return text


def test_graph(model, tests):
    family = detect_family(model)
    opts = get_options(model)
    graph_sys = GRAPH_SYS + (GRAPH_35_SUFFIX if family == 'qwen3.5' else '')

    # Unload all
    ps = requests.get(f'{OLLAMA}/api/ps').json()
    for m in ps.get('models', []):
        requests.post(f'{OLLAMA}/api/generate', json={'model': m['name'], 'keep_alive': 0})
    time.sleep(2)

    # Warm up
    requests.post(f'{OLLAMA}/api/chat', json={
        'model': model, 'messages': [{'role': 'user', 'content': 'Ready'}],
        'stream': False, 'think': False,
    }, timeout=120)

    sep = '=' * 60
    print(f'\n{sep}')
    print(f'MODEL: {model} (family={family})')
    print(f'Options: {opts}')
    print(f'Graph suffix: {"YES" if family == "qwen3.5" else "NO"}')
    print(sep)

    total_e, total_r, total_t = 0, 0, 0.0
    for test_name, text in tests.items():
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
        elapsed = time.time() - start
        raw = r.json()['message']['content']
        try:
            g = json.loads(strip_fences(raw))
            ents = g.get('entities', [])
            rels = g.get('relationships', [])
        except:
            ents, rels = [], []

        total_e += len(ents)
        total_r += len(rels)
        total_t += elapsed

        print(f'\n  [{test_name}] {len(ents)}e/{len(rels)}r ({elapsed:.1f}s)')
        for e in ents[:8]:
            print(f'    E: {e.get("name","?")} ({e.get("type","?")})')
        for rel in rels[:10]:
            print(f'    R: {rel.get("source","?")} --{rel.get("relation","?")}--> {rel.get("target","?")}')
        if not rels:
            print(f'    NO RELATIONSHIPS')

    print(f'\n  TOTAL: {total_e}e/{total_r}r in {total_t:.1f}s')
    return total_e, total_r, total_t


if __name__ == '__main__':
    results = {}
    for model in ['qwen3:4b-instruct-2507-q4_K_M', 'qwen3.5:4b']:
        e, r, t = test_graph(model, TESTS)
        results[model] = (e, r, t)

    print(f'\n{"=" * 60}')
    print('COMPARISON')
    print(f'{"=" * 60}')
    for model, (e, r, t) in results.items():
        short = model.split(':')[0]
        print(f'  {short:>10}: {e}e / {r}r / {t:.1f}s')
