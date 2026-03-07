#!/usr/bin/env python3
"""Test extraction through the production filter pipeline (not just raw LLM output).
Simulates what fix_graph_entity_parsing.py actually does with the LLM output."""
import requests, json, time, re

OLLAMA = 'http://ollama:11434'

# --- Copy of production filters from fix_graph_entity_parsing.py ---

_ENTITY_BLOCKLIST = frozenset([
    'php', 'js', 'jsx', 'ts', 'tsx', 'py', 'json', 'yaml', 'yml', 'css', 'html',
    'xml', 'md', 'txt', 'sh', 'bat', 'sql', 'csv', 'log', 'conf', 'cfg', 'ini',
    'utilities', 'default', 'record', 'entity', 'unknown', 'other', 'none', 'null',
    'true', 'false', 'yes', 'no', 'n/a', 'etc',
    'metadata', 'project', 'timezone', 'port', 'api', 'url', 'host',
])

_NUMBER_PATTERN = re.compile(
    r'^[\d$\xe2\x82\xac\xc2\xa3\xc2\xa5,.%]+$'
    r'|^\d+[\w_]*$'
    r'|^\d+[_-]\w+'
    r'|^\d+:\d+'
    r'|\d+_(months?|years?|days?|hours?|minutes?|seconds?|euros?|dollars?|fps|gb|mb|tb|gbe)$'
)

_BLOCKED_ENTITY_TYPES = frozenset(['concept', 'metric'])

# UPDATED: includes game_element
_HUB_TYPES = frozenset(['person', 'project', 'game_element'])


def detect_family(model):
    name = model.lower()
    if 'qwen3.5' in name or 'qwen35' in name: return 'qwen3.5'
    if 'qwen3' in name: return 'qwen3'
    return 'unknown'


def get_options(model):
    if detect_family(model) == 'qwen3.5':
        return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}
    return {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20}


def strip_fences(text):
    text = text.strip()
    if text.startswith('```'):
        nl = text.find('\n')
        if nl != -1: text = text[nl + 1:]
        if text.rstrip().endswith('```'): text = text.rstrip()[:-3].rstrip()
    return text


def apply_production_filters(parsed, user_id='steven'):
    """Apply the same filters as fix_graph_entity_parsing._json_extract_graph"""
    _SELF_REFS = frozenset(['i', 'me', 'my', 'myself', 'user', 'the_user', 'the_subject', 'subject'])
    user_id_key = user_id.lower().replace(' ', '_')

    def _norm(name_key):
        return user_id_key if name_key in _SELF_REFS else name_key

    entity_type_map = {}
    for ent in parsed.get('entities', []):
        if isinstance(ent, dict):
            name = ent.get('name', '').strip()
            etype = ent.get('type', 'concept').strip()
        else:
            continue
        if not name:
            continue
        name_key = _norm(name.lower().replace(' ', '_'))
        etype_key = etype.lower().replace(' ', '_')
        if name_key == user_id_key:
            etype_key = 'person'
        if name_key in _ENTITY_BLOCKLIST: continue
        if len(name_key) < 2: continue
        if len(name_key) > 60: continue
        if _NUMBER_PATTERN.match(name_key): continue
        if etype_key in _BLOCKED_ENTITY_TYPES: continue
        entity_type_map[name_key] = etype_key

    relationships = []
    filtered_reasons = []
    for rel in parsed.get('relationships', []):
        if not isinstance(rel, dict): continue
        source = _norm(str(rel.get('source', '')).strip().lower().replace(' ', '_'))
        target = _norm(str(rel.get('target', '')).strip().lower().replace(' ', '_'))
        relation = str(rel.get('relation', '')).strip().lower().replace(' ', '_')
        if not source or not target or not relation: continue
        if source == target:
            filtered_reasons.append(f"self-ref: {source}")
            continue
        if len(source) > 2 and len(target) > 2 and (source in target or target in source):
            filtered_reasons.append(f"fuzzy-self: {source}->{target}")
            continue
        if source in _ENTITY_BLOCKLIST or target in _ENTITY_BLOCKLIST: continue
        if _NUMBER_PATTERN.match(source) or _NUMBER_PATTERN.match(target): continue

        user_id_key = user_id.lower().replace(' ', '_')
        if source not in entity_type_map:
            if source == user_id_key:
                entity_type_map[source] = 'person'
            else:
                filtered_reasons.append(f"unknown-source: {source}")
                continue
        if target not in entity_type_map:
            if target == user_id_key:
                entity_type_map[target] = 'person'
            else:
                filtered_reasons.append(f"unknown-target: {target}")
                continue

        source_type = entity_type_map.get(source, '')
        if source_type not in _HUB_TYPES:
            filtered_reasons.append(f"non-hub: {source}({source_type})->{target}")
            continue

        relationships.append({
            'source': source,
            'relationship': relation,
            'destination': target,
        })

    return entity_type_map, relationships, filtered_reasons


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
    "emotional_decisions": """I spent 3 days debugging why Ollama's tool calling fails with qwen3 models. Incredibly frustrating.
The root cause is that qwen3:4b returns tool call syntax as plain text instead of structured tool_calls objects.
I ended up replacing the entire tool-calling pipeline with JSON mode extraction, which works perfectly.
Honestly the JSON approach is better anyway - single LLM call instead of 3-4 sequential tool calls, and the output is more consistent.
Considering switching the embedding model from nomic-embed-text to qwen3-embedding:0.6b for better multilingual support.
Already tested it - same quality for English facts, much better for Estonian and Russian text, and 5.7x faster.""",
}

if __name__ == '__main__':
    for model in ['qwen3:4b-instruct-2507-q4_K_M', 'qwen3.5:4b']:
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

        print(f'\n{"=" * 70}')
        print(f'MODEL: {model} (family={family})')
        print(f'Options: {opts}')
        print(f'{"=" * 70}')

        total_raw_r, total_filtered_r, total_t = 0, 0, 0.0
        for test_name, text in TESTS.items():
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
                parsed = json.loads(strip_fences(raw))
            except:
                parsed = {'entities': [], 'relationships': []}

            raw_ents = parsed.get('entities', [])
            raw_rels = parsed.get('relationships', [])

            # Apply production filters
            ent_map, filtered_rels, reasons = apply_production_filters(parsed)

            total_raw_r += len(raw_rels)
            total_filtered_r += len(filtered_rels)
            total_t += elapsed

            print(f'\n  [{test_name}] raw={len(raw_ents)}e/{len(raw_rels)}r -> filtered={len(ent_map)}e/{len(filtered_rels)}r ({elapsed:.1f}s)')
            for rel in filtered_rels[:8]:
                src_type = ent_map.get(rel['source'], '?')
                print(f'    R: {rel["source"]}({src_type}) --{rel["relationship"]}--> {rel["destination"]}')
            if reasons:
                print(f'    FILTERED: {", ".join(reasons[:5])}')

        print(f'\n  TOTAL: raw {total_raw_r}r -> filtered {total_filtered_r}r in {total_t:.1f}s')
        loss_pct = ((total_raw_r - total_filtered_r) / total_raw_r * 100) if total_raw_r > 0 else 0
        print(f'  Filter loss: {loss_pct:.0f}%')
