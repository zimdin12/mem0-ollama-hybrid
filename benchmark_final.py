#!/usr/bin/env python3
"""Final comprehensive benchmark with production prompt + filters + all fixes."""
import requests, json, time, re, os, sys

OLLAMA = 'http://ollama:11434'

# Import production code if available, otherwise inline
try:
    sys.path.insert(0, '/usr/src/openmemory')
    from fix_graph_entity_parsing import (
        _detect_model_family, _get_llm_options, _strip_json_fences,
        GRAPH_EXTRACTION_PROMPT, _ENTITY_BLOCKLIST, _NUMBER_PATTERN,
        _BLOCKED_ENTITY_TYPES, _HUB_TYPES,
    )
    USING_PRODUCTION = True
except ImportError:
    USING_PRODUCTION = False
    print("WARNING: Could not import production code, using inline fallbacks")

_SELF_REFS = frozenset(['i', 'me', 'my', 'myself', 'user', 'the_user', 'the_subject', 'subject'])


def strip_fences(text):
    text = text.strip()
    if text.startswith('```'):
        nl = text.find('\n')
        if nl != -1: text = text[nl + 1:]
        if text.rstrip().endswith('```'): text = text.rstrip()[:-3].rstrip()
    return text


def apply_filters(parsed, user_id='steven'):
    """Production-equivalent filtering."""
    uid = user_id.lower().replace(' ', '_')
    def norm(n):
        return uid if n in _SELF_REFS else n

    emap = {}
    for ent in parsed.get('entities', []):
        if not isinstance(ent, dict): continue
        name = ent.get('name', '').strip()
        etype = ent.get('type', 'concept').strip()
        if not name: continue
        nk = norm(name.lower().replace(' ', '_'))
        ek = etype.lower().replace(' ', '_')
        if nk == uid: ek = 'person'
        if nk in _ENTITY_BLOCKLIST or len(nk) < 2 or len(nk) > 60: continue
        if _NUMBER_PATTERN.match(nk) or ek in _BLOCKED_ENTITY_TYPES: continue
        emap[nk] = ek

    rels = []
    filtered = []
    for rel in parsed.get('relationships', []):
        if not isinstance(rel, dict): continue
        s = norm(str(rel.get('source', '')).strip().lower().replace(' ', '_'))
        t = norm(str(rel.get('target', '')).strip().lower().replace(' ', '_'))
        r = str(rel.get('relation', '')).strip().lower().replace(' ', '_')
        if not s or not t or not r: continue
        if s == t:
            filtered.append(f'self:{s}')
            continue
        if len(s) > 2 and len(t) > 2 and (s in t or t in s):
            filtered.append(f'fuzzy:{s}->{t}')
            continue
        if s in _ENTITY_BLOCKLIST or t in _ENTITY_BLOCKLIST: continue
        if _NUMBER_PATTERN.match(s) or _NUMBER_PATTERN.match(t): continue
        if s not in emap:
            if s == uid: emap[s] = 'person'
            else:
                filtered.append(f'unk-src:{s}')
                continue
        if t not in emap:
            if t == uid: emap[t] = 'person'
            else:
                filtered.append(f'unk-tgt:{t}')
                continue
        if emap.get(s, '') not in _HUB_TYPES:
            filtered.append(f'non-hub:{s}({emap.get(s,"")})')
            continue
        rels.append({'source': s, 'rel': r, 'target': t})
    return emap, rels, filtered


TESTS = {
    # --- Basic ---
    "basic_facts": {
        "input": "Steven lives in Tallinn, Estonia. He has a cat named Pixel. His birthday is March 15th.",
        "check": "3 entities (person, place, animal), 2+ relationships",
    },
    # --- Negation ---
    "negation": {
        "input": "Steven does NOT use Visual Studio Code anymore. He switched from VS Code to Neovim last month. He tried Emacs but didn't like it. He has never used IntelliJ.",
        "check": "does_not_use/stopped_using for VS Code, switched_to Neovim, tried/disliked Emacs, never_used IntelliJ",
    },
    # --- Temporal ---
    "temporal": {
        "input": "Steven used to work at Google but left in 2024. He now works at a startup called NordicAI. Before Google he was at Microsoft for 2 years.",
        "check": "formerly/previously for Google+Microsoft, currently for NordicAI",
    },
    # --- Complex project ---
    "complex_project": {
        "input": """Echoes of the Fallen is a voxel roguelike built with Unreal Engine 5 and C++.
The rendering pipeline uses Voxel Plugin 2.0 which leverages UE5's Nanite for virtualized geometry and Lumen for global illumination.
Mesh generation uses dual contouring instead of marching cubes because it preserves sharp edges for architectural ruins.
The chunk system is 32x32x32 with octree LOD, greedy meshing for 40% triangle reduction, and async generation on background threads.
Steven chose this tech stack after evaluating Unity with Cubiquity and Godot with VoxelTools, rejecting both due to limited LOD support.""",
        "check": "project->built_with UE5+C++, project->uses dual_contouring, rejected Unity+Godot",
    },
    # --- First person ---
    "first_person": {
        "input": "I built a memory system using Qdrant and Neo4j. I replaced tool calling with JSON mode because it was unreliable. My favorite editor is Neovim.",
        "check": "steven->built/uses qdrant+neo4j, steven->uses json_mode, steven->uses neovim",
    },
    # --- Preferences with negation ---
    "preferences": {
        "input": "Steven prefers Arch Linux over Ubuntu. He dislikes Windows but uses it for gaming. He refuses to use macOS.",
        "check": "prefers Arch, dislikes Windows, uses Windows (for gaming), refuses macOS",
    },
    # --- People and roles ---
    "people_roles": {
        "input": """Steven, Maria, and Jake are building a game together. Steven handles all C++ gameplay programming and Unreal blueprints.
Maria is the art director who creates voxel assets in MagicaVoxel and does character animation in Blender.
Jake manages the CI/CD pipeline using GitHub Actions and Docker, and also handles QA testing on both Windows and Linux.
Maria reports to Steven as project lead but has full creative control over the art direction.
Jake previously worked at Ubisoft Montreal for 3 years before joining as a freelancer.""",
        "check": "3 people, tool associations, reporting structure, Jake's history",
    },
    # --- Corrections/updates ---
    "corrections": {
        "input": """The game was originally called 'Fallen Echoes' but Steven renamed it to 'Echoes of the Fallen' because the original name was too similar to an existing mobile game.
Development started in January 2025 with a 12-month timeline, but after adding multiplayer architecture the scope expanded to 18 months with a target release of July 2026.
Steven first planned Steam-only release but added Epic Games Store after learning about their 12% revenue split versus Steam's 30%.""",
        "check": "renamed relationship, Steam+Epic, captures WHY",
    },
    # --- Emotional/decisions ---
    "decisions": {
        "input": """I spent 3 days debugging why Ollama's tool calling fails with qwen3 models. Incredibly frustrating.
The root cause is that qwen3:4b returns tool call syntax as plain text instead of structured tool_calls objects.
I ended up replacing the entire tool-calling pipeline with JSON mode extraction, which works perfectly.
Considering switching the embedding model from nomic-embed-text to qwen3-embedding:0.6b for better multilingual support.""",
        "check": "steven->uses json_mode, steven->considering qwen3-embedding",
    },
    # --- Mixed languages ---
    "mixed_lang": {
        "input": """Steven codes primarily in English but writes commit messages mixing Estonian and English.
His variable naming convention is camelCase for C++ (following UE5 style) and snake_case for Python and Rust.
He uses 'tere' (Estonian for hello) as his test greeting in all language-related unit tests.""",
        "check": "steven->uses C++/Python/Rust, coding conventions",
    },
    # --- Skill system (ambiguous refs) ---
    "skill_system": {
        "input": """Steven's game uses a skill system inspired by isekai anime where death triggers skill evolution.
When the player dies, fresh skills from that run become permanent - they call this the 'ripening' mechanic.
Swimming and water breathing combine into deep diving - a hybrid skill that unlocks underwater cave exploration.
Steven says it's the core differentiator from competitors like Hades and Dead Cells which use item-based progression.""",
        "check": "game connects to ripening mechanic, differentiates from Hades/Dead Cells",
    },
}


def run_model(model, user_id='steven'):
    family = _detect_model_family(model)
    options = _get_llm_options(family)
    short = model.split(':')[0]

    # Build production prompt
    prompt = GRAPH_EXTRACTION_PROMPT.replace("USER_ID", user_id)
    if family == 'qwen3.5':
        prompt += (
            "\n\nIMPORTANT: You MUST generate relationships for every entity. "
            "Every entity should connect to at least one person or project via a relationship. "
            "If an entity has no relationship, remove it from entities too."
            "\nPreserve NEGATIONS: does_not_use, stopped_using, rejected, never_used (NOT just 'uses')."
            "\nPreserve TEMPORAL state: formerly_worked_at vs currently_works_at, previously_used vs uses."
        )

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

    print(f'\n{"=" * 70}')
    print(f'MODEL: {model} (family={family})')
    print(f'Options: {options}')
    print(f'Prompt suffix: {"YES (rel+neg+temporal)" if family == "qwen3.5" else "NO"}')
    print(f'{"=" * 70}')

    totals = {'raw_e': 0, 'raw_r': 0, 'filt_e': 0, 'filt_r': 0, 'time': 0.0}
    scores = {'negation_ok': 0, 'negation_total': 0, 'self_ref_ok': 0, 'self_ref_total': 0}

    for test_name, test in TESTS.items():
        text = test['input']

        start = time.time()
        r = requests.post(f'{OLLAMA}/api/chat', json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': f'Extract from:\n{text}'},
            ],
            'stream': False, 'format': 'json', 'think': False,
            'options': options,
        }, timeout=120)
        elapsed = time.time() - start

        content = r.json().get('message', {}).get('content', '')
        try:
            parsed = json.loads(strip_fences(content))
        except:
            parsed = {'entities': [], 'relationships': []}

        raw_e = len(parsed.get('entities', []))
        raw_r = len(parsed.get('relationships', []))
        emap, frels, filtered = apply_filters(parsed, user_id)

        totals['raw_e'] += raw_e
        totals['raw_r'] += raw_r
        totals['filt_e'] += len(emap)
        totals['filt_r'] += len(frels)
        totals['time'] += elapsed

        # Quality scoring
        if test_name == 'negation':
            for fr in frels:
                scores['negation_total'] += 1
                rel_name = fr['rel']
                tgt = fr['target']
                # VS Code should have negative relation
                if 'visual_studio' in tgt or 'vs_code' in tgt:
                    if 'not' in rel_name or 'stop' in rel_name or 'switch' in rel_name:
                        scores['negation_ok'] += 1
                    else:
                        scores['negation_ok'] += 0  # wrong
                else:
                    scores['negation_ok'] += 1  # other rels are fine

        if test_name in ('first_person', 'decisions'):
            scores['self_ref_total'] += 1
            if any(fr['source'] == user_id for fr in frels):
                scores['self_ref_ok'] += 1

        print(f'\n  [{test_name}] raw={raw_e}e/{raw_r}r -> filt={len(emap)}e/{len(frels)}r ({elapsed:.1f}s)')
        for fr in frels[:10]:
            src_type = emap.get(fr['source'], '?')
            print(f'    {fr["source"]}({src_type}) --{fr["rel"]}--> {fr["target"]}')
        if filtered:
            print(f'    FILTERED({len(filtered)}): {", ".join(filtered[:4])}')

    print(f'\n  {"─" * 50}')
    print(f'  TOTALS: raw={totals["raw_e"]}e/{totals["raw_r"]}r -> filt={totals["filt_e"]}e/{totals["filt_r"]}r')
    loss = ((totals['raw_r'] - totals['filt_r']) / totals['raw_r'] * 100) if totals['raw_r'] > 0 else 0
    print(f'  Filter loss: {loss:.0f}%')
    print(f'  Total time: {totals["time"]:.1f}s (avg {totals["time"]/len(TESTS):.1f}s/test)')
    print(f'  Negation accuracy: {scores["negation_ok"]}/{scores["negation_total"]}')
    print(f'  Self-ref (I->user): {scores["self_ref_ok"]}/{scores["self_ref_total"]}')

    return totals, scores


if __name__ == '__main__':
    models = ['qwen3:4b-instruct-2507-q4_K_M', 'qwen3.5:4b']

    # Check if 9b is available
    try:
        r = requests.get(f'{OLLAMA}/api/tags')
        available = [m['name'] for m in r.json().get('models', [])]
        if any('qwen3.5:9b' in m for m in available):
            models.append('qwen3.5:9b')
            print("qwen3.5:9b available, including in test")
    except:
        pass

    all_results = {}
    for model in models:
        totals, scores = run_model(model)
        all_results[model] = (totals, scores)

    # Summary comparison
    print(f'\n\n{"=" * 70}')
    print('FINAL COMPARISON')
    print(f'{"=" * 70}')
    print(f'{"Model":<35} {"Entities":>8} {"Rels":>6} {"Loss":>6} {"Time":>7} {"Neg":>5} {"I->u":>5}')
    print('─' * 70)
    for model in models:
        t, s = all_results[model]
        short = model[:34]
        loss = ((t['raw_r'] - t['filt_r']) / t['raw_r'] * 100) if t['raw_r'] > 0 else 0
        neg = f'{s["negation_ok"]}/{s["negation_total"]}'
        imu = f'{s["self_ref_ok"]}/{s["self_ref_total"]}'
        print(f'{short:<35} {t["filt_e"]:>8} {t["filt_r"]:>6} {loss:>5.0f}% {t["time"]:>6.1f}s {neg:>5} {imu:>5}')
