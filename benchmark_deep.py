#!/usr/bin/env python3
"""Deep extraction benchmark: 20 test cases across all difficulty levels.
Tests raw graph + fact extraction quality, timing, and edge cases."""
import requests, json, time, sys, os

sys.path.insert(0, '/usr/src/openmemory')
from fix_graph_entity_parsing import (
    GRAPH_EXTRACTION_PROMPT, _detect_model_family, _get_llm_options,
    _strip_json_fences, _ENTITY_BLOCKLIST, _NUMBER_PATTERN,
    _BLOCKED_ENTITY_TYPES, _HUB_TYPES,
)

OLLAMA = 'http://ollama:11434'
_SELF_REFS = frozenset(['i', 'me', 'my', 'myself', 'user', 'the_user', 'the_subject', 'subject'])

FACT_SYSTEM = """Extract all factual information from the text. Return each fact on a new line.
Rules:
1. Each fact must be a complete, standalone sentence with the subject named.
2. Preserve negations (does NOT, never, stopped, rejected).
3. Preserve reasons and conditions (because, due to, after, instead of).
4. Preserve numbers, measurements, percentages, and thresholds exactly.
5. Do NOT make up information not in the text.
6. Do NOT add explanations or commentary.
7. Replace "I/me/my" with the person's name if identifiable from context.
8. If the text contains NO factual information worth remembering, return exactly: NONE"""


# ============================================================
# TEST CASES — 20 scenarios across all difficulty levels
# ============================================================

TESTS = {
    # ---- TRIVIAL (1 fact) ----
    "trivial_1fact": {
        "input": "Steven likes pizza.",
        "expect_facts": 1,
        "expect_rels": 1,
        "notes": "Simplest possible. Should produce 1 fact, 1 relationship.",
    },

    # ---- SHORT (2-3 facts) ----
    "short_basic": {
        "input": "Steven is 28 years old and lives in Tallinn. He works as a game developer.",
        "expect_facts": 3,
        "expect_rels": 2,
        "notes": "Basic bio. Age, location, job.",
    },

    # ---- ALMOST NO FACTS ----
    "noise_filler": {
        "input": "It was a pretty good day today. I woke up, had coffee, and went for a walk. Nothing special happened. The weather was nice though.",
        "expect_facts": 0,
        "expect_rels": 0,
        "notes": "Daily routine noise. Should extract ZERO facts. Any extraction is hallucination.",
    },
    "noise_greeting": {
        "input": "Hey! How's it going? Just wanted to check in and see how you're doing.",
        "expect_facts": 0,
        "expect_rels": 0,
        "notes": "Pure greeting. Should extract ZERO facts.",
    },
    "noise_instruction": {
        "input": "Please search for my memories about Python. I want to see what you know about my coding preferences.",
        "expect_facts": 0,
        "expect_rels": 0,
        "notes": "Meta-instruction. Should extract ZERO facts (this is a command, not information).",
    },

    # ---- MEDIUM (5-8 facts) ----
    "medium_preferences": {
        "input": "Steven uses Neovim as his primary editor with the Lazy plugin manager. He prefers dark themes, specifically Tokyo Night. His terminal is Alacritty on Linux and Windows Terminal on Windows. He types at about 95 WPM.",
        "expect_facts": 5,
        "expect_rels": 4,
        "notes": "Preferences and tools. Specific details matter.",
    },
    "medium_negation": {
        "input": "Steven does NOT use Docker Desktop — he runs Docker Engine directly on Linux. He tried Podman but went back to Docker because Podman had compatibility issues with his compose files. He refuses to use Snap packages.",
        "expect_facts": 4,
        "expect_rels": 3,
        "notes": "Multiple negations and reasons. Must preserve NOT, tried-but-rejected, refuses.",
    },

    # ---- FIRST PERSON ----
    "first_person_tech": {
        "input": "I've been using Rust for about 6 months now. I switched from C++ because I was tired of memory bugs. My main project is a voxel engine called CrystalForge. I deploy it using GitHub Actions to a Hetzner VPS.",
        "expect_facts": 4,
        "expect_rels": 4,
        "notes": "All first person. Must replace I/my with steven. Must capture project name and deployment.",
    },

    # ---- COMPLEX MULTI-ENTITY ----
    "multi_person_4": {
        "input": """Steven leads the development of Echoes of the Fallen. Maria does all the 3D art using Blender and MagicaVoxel.
Jake handles DevOps with GitHub Actions and also does QA testing. Lisa joined recently as the sound designer, using FMOD and Reaper.
Maria reports to Steven. Jake and Lisa both report to Maria for task assignments but to Steven for technical decisions.
Jake previously worked at Epic Games for 4 years. Lisa is a freelancer who also works with two other indie studios.""",
        "expect_facts": 10,
        "expect_rels": 12,
        "notes": "4 people. Complex reporting. Must NOT cross-contaminate (Jake's Epic != Lisa's Epic, etc).",
    },

    # ---- TEMPORAL CHANGES ----
    "temporal_career": {
        "input": "Steven started coding in PHP in 2015. He moved to Python in 2018 for machine learning work. In 2022 he picked up Rust and now uses it as his primary language. He still uses Python for scripting but no longer writes PHP at all.",
        "expect_facts": 5,
        "expect_rels": 4,
        "notes": "Career progression. Must distinguish past vs present. 'no longer' = stopped.",
    },

    # ---- CONFLICTING / CORRECTIVE ----
    "conflicting_updates": {
        "input": """The project budget was initially $2,000 but was revised to $5,000 after scope expansion.
The release date was set for March 2026 but has been pushed to September 2026.
The team size was 3 but grew to 5 with the addition of Lisa and Tom.
Originally targeting only PC but now also planning console ports for PS5 and Xbox Series X.""",
        "expect_facts": 8,
        "expect_rels": 2,
        "notes": "Old→New updates. Must capture CURRENT values. Should NOT create entities from numbers/dates.",
    },

    # ---- CHAINING (A→B→C→D) ----
    "chain_tech": {
        "input": """Steven built CrystalForge using Rust. CrystalForge uses wgpu for rendering, which is a Rust wrapper around WebGPU.
The rendering pipeline feeds into a custom voxel mesher that outputs to wgpu's render pass system.
The mesher was inspired by Transvoxel algorithm but modified for speed. The modifications were published as a crate called fast-transvoxel on crates.io.""",
        "expect_facts": 6,
        "expect_rels": 6,
        "notes": "Chain: Steven→CrystalForge→wgpu→WebGPU, plus mesher→Transvoxel→fast-transvoxel. Tests deep linking.",
    },

    # ---- DENSE TECHNICAL ----
    "dense_technical": {
        "input": """The CI/CD pipeline uses GitHub Actions with 3 workflows: build (runs on ubuntu-latest with Rust 1.78),
test (parallel matrix across Windows/Linux/macOS with coverage via tarpaulin),
and deploy (pushes Docker images to ghcr.io tagged with git SHA, then triggers ArgoCD sync to the k8s cluster).
Build artifacts are cached using sccache backed by S3. Average build time is 4 minutes, down from 12 after adding incremental compilation.
The k8s cluster runs on 3 Hetzner CX32 nodes with Cilium CNI and Longhorn storage.""",
        "expect_facts": 10,
        "expect_rels": 5,
        "notes": "Very dense infra. Many numbers. Should extract tech entities, not numbers as entities.",
    },

    # ---- EMOTIONAL / OPINIONATED ----
    "emotional_rant": {
        "input": """I absolutely HATE the JavaScript ecosystem. Every time I start a project, half the dependencies are deprecated.
NPM is a security nightmare — I've had 3 supply chain scares this year alone.
That said, I have to admit TypeScript is actually pretty good. I use it for all my web projects now.
Deno is interesting but the ecosystem is too small. Bun is fast but unstable. Node.js is the devil I know.""",
        "expect_facts": 5,
        "expect_rels": 5,
        "notes": "Strong opinions. Must capture: hates JS ecosystem, likes TypeScript, uses TS for web, opinions on Deno/Bun/Node.",
    },

    # ---- MIXED LANGUAGES ----
    "mixed_estonian": {
        "input": """Steven's project documentation is in English but his personal notes are in Estonian.
He calls his test database 'proov_andmebaas' (Estonian for test database).
The error messages use i18n with English as default and Estonian as fallback.
His git branch naming uses English: feature/, bugfix/, hotfix/.""",
        "expect_facts": 4,
        "expect_rels": 2,
        "notes": "Bilingual content. Must preserve Estonian terms with context.",
    },

    # ---- VERY LONG (paragraph) ----
    "long_project_desc": {
        "input": """Echoes of the Fallen is a single-player voxel roguelike built with Unreal Engine 5 and C++. The game features a unique death mechanic called 'ripening' where skills gained during a run become permanent upon death. The world is procedurally generated using dual contouring with 32x32x32 chunks and octree LOD.

The rendering pipeline uses Voxel Plugin 2.0 which leverages UE5's Nanite for virtualized geometry and Lumen for global illumination. Character models are hand-crafted voxel art created by Maria in MagicaVoxel, then rigged and animated in Blender.

The game targets Steam and Epic Games Store with a release date of September 2026. Development started in January 2025 with a budget that grew from $500 to $5,000. The team consists of Steven (lead developer, C++ gameplay), Maria (art director), Jake (DevOps and QA), and Lisa (sound design with FMOD).

Steven chose Unreal over Unity (poor Linux support) and Godot (limited LOD). The multiplayer architecture was added mid-development, extending the timeline from 12 to 18 months. The skill system is inspired by isekai anime and differentiates from Hades and Dead Cells by using skill evolution instead of item-based progression.""",
        "expect_facts": 20,
        "expect_rels": 15,
        "notes": "Full project doc. Must capture project details, team roles, tech stack, timeline, comparisons.",
    },

    # ---- AMBIGUOUS PRONOUNS ----
    "ambiguous_pronouns": {
        "input": "Steven and Jake both use Docker. He prefers the CLI while he likes Docker Desktop. Steven showed Jake how to write multi-stage Dockerfiles. Jake then taught the rest of the team.",
        "expect_facts": 4,
        "expect_rels": 3,
        "notes": "Ambiguous 'he/he'. Model should either resolve correctly or skip ambiguous parts.",
    },

    # ---- NUMBERS AND SPECS ----
    "numbers_heavy": {
        "input": "The server has 128GB RAM, 2x AMD EPYC 7543 CPUs (32 cores each), 4x NVIDIA A100 80GB GPUs, and 8TB NVMe storage in RAID 10. Network is 25Gbps with a 10Gbps dedicated backup link. Power consumption averages 3.2kW.",
        "expect_facts": 6,
        "expect_rels": 1,
        "notes": "Hardware specs. Numbers should be in FACTS not graph entities. Graph should have 'server' entity.",
    },

    # ---- IMPLICIT FACTS ----
    "implicit_knowledge": {
        "input": "After mass adoption of Claude 4.5, Steven's startup pivoted from building custom fine-tuned models to building AI agent tooling instead. The pivot made sense — why compete with Anthropic when you can build on top of their API?",
        "expect_facts": 3,
        "expect_rels": 3,
        "notes": "Business decision with implicit reasoning. Should capture: pivot, why, what from/to.",
    },

    # ---- CONTRADICTORY IN SAME TEXT ----
    "contradiction": {
        "input": "Steven's favorite language is Rust. Actually, scratch that — he's been using Go more lately and thinks it might be his new favorite. Though honestly, for quick scripts he still reaches for Python every time.",
        "expect_facts": 3,
        "expect_rels": 3,
        "notes": "Self-correcting. Should capture the FINAL state: Go might be favorite, Rust was favorite, Python for scripts.",
    },
}


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

    rels, filtered = [], []
    for rel in parsed.get('relationships', []):
        if not isinstance(rel, dict): continue
        s = norm(str(rel.get('source', '')).strip().lower().replace(' ', '_'))
        t = norm(str(rel.get('target', '')).strip().lower().replace(' ', '_'))
        r = str(rel.get('relation', '')).strip().lower().replace(' ', '_')
        if not s or not t or not r: continue
        if s == t: filtered.append(f'self:{s}'); continue
        if len(s) > 2 and len(t) > 2 and (s in t or t in s): filtered.append(f'fuzz:{s}->{t}'); continue
        if s in _ENTITY_BLOCKLIST or t in _ENTITY_BLOCKLIST: continue
        if _NUMBER_PATTERN.match(s) or _NUMBER_PATTERN.match(t): continue
        if s not in emap:
            if s == uid: emap[s] = 'person'
            else: filtered.append(f'unk-s:{s}'); continue
        if t not in emap:
            if t == uid: emap[t] = 'person'
            else: filtered.append(f'unk-t:{t}'); continue
        if emap.get(s, '') not in _HUB_TYPES: filtered.append(f'hub:{s}({emap.get(s,"")})'); continue
        rels.append({'s': s, 'r': r, 't': t})
    return emap, rels, filtered


def extract_facts(model, text, options):
    """Extract facts via LLM."""
    r = requests.post(f'{OLLAMA}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': FACT_SYSTEM},
            {'role': 'user', 'content': f'Extract facts from:\n{text}'},
        ],
        'stream': False, 'think': False, 'options': options,
    }, timeout=120)
    content = r.json()['message']['content']
    # Check for NONE response
    if content.strip().upper() == 'NONE' or content.strip() == '':
        return [], 0
    lines = [l.strip() for l in content.split('\n')
             if l.strip() and len(l.strip()) > 15
             and not l.strip().startswith('#')
             and not l.strip().startswith('|')
             and '**:' not in l.strip()
             and 'extracted' not in l.strip().lower()[:20]
             and l.strip().upper() != 'NONE']
    # Strip numbering
    cleaned = []
    for l in lines:
        l = l.lstrip('0123456789.-) *•')
        if len(l.strip()) > 15:
            cleaned.append(l.strip())
    return cleaned, r.json().get('eval_count', 0)


def extract_graph(model, text, prompt, options):
    """Extract graph via LLM with production filters."""
    r = requests.post(f'{OLLAMA}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'Extract from:\n{text}'},
        ],
        'stream': False, 'format': 'json', 'think': False, 'options': options,
    }, timeout=120)
    content = r.json()['message']['content']
    try:
        parsed = json.loads(_strip_json_fences(content))
    except:
        return {}, [], [], 0
    emap, rels, filtered = apply_filters(parsed)
    return emap, rels, filtered, r.json().get('eval_count', 0)


def run_model(model):
    family = _detect_model_family(model)
    options = _get_llm_options(family)
    short = model.split(':')[0]

    # Build graph prompt
    gprompt = GRAPH_EXTRACTION_PROMPT.replace("USER_ID", "steven")
    if family == 'qwen3.5':
        gprompt += (
            "\n\nIMPORTANT: You MUST generate relationships for every entity. "
            "Every entity should connect to at least one person or project via a relationship. "
            "If an entity has no relationship, remove it from entities too."
            "\nPreserve NEGATIONS: does_not_use, stopped_using, rejected, never_used (NOT just 'uses')."
            "\nPreserve TEMPORAL state: formerly_worked_at vs currently_works_at, previously_used vs uses."
        )

    # Unload all models
    ps = requests.get(f'{OLLAMA}/api/ps').json()
    for m in ps.get('models', []):
        requests.post(f'{OLLAMA}/api/generate', json={'model': m['name'], 'keep_alive': 0})
    time.sleep(5)

    # Warm up with 2 calls
    for _ in range(2):
        requests.post(f'{OLLAMA}/api/chat', json={
            'model': model, 'messages': [{'role': 'user', 'content': 'Ready for extraction.'}],
            'stream': False, 'think': False,
        }, timeout=120)
    time.sleep(1)

    sep = '=' * 75
    print(f'\n{sep}')
    print(f'MODEL: {model} (family={family})')
    print(f'Options: {options}')
    print(sep)

    results = {}
    totals = {'facts': 0, 'ents': 0, 'rels': 0, 'filtered': 0,
              'fact_time': 0, 'graph_time': 0, 'fact_tokens': 0, 'graph_tokens': 0}
    quality = {'noise_false_pos': 0, 'noise_tests': 0,
               'negation_correct': 0, 'negation_total': 0,
               'selfref_correct': 0, 'selfref_total': 0}

    for test_name, test in TESTS.items():
        text = test['input']
        is_noise = test['expect_facts'] == 0

        # Fact extraction
        t0 = time.time()
        facts, fact_tokens = extract_facts(model, text, options)
        fact_time = time.time() - t0

        # Graph extraction
        t0 = time.time()
        emap, rels, filtered, graph_tokens = extract_graph(model, text, gprompt, options)
        graph_time = time.time() - t0

        totals['facts'] += len(facts)
        totals['ents'] += len(emap)
        totals['rels'] += len(rels)
        totals['filtered'] += len(filtered)
        totals['fact_time'] += fact_time
        totals['graph_time'] += graph_time
        totals['fact_tokens'] += fact_tokens
        totals['graph_tokens'] += graph_tokens

        # Quality scoring
        if is_noise:
            quality['noise_tests'] += 1
            if len(facts) > 0 or len(rels) > 0:
                quality['noise_false_pos'] += 1

        if test_name in ('medium_negation', 'temporal_career'):
            for rel in rels:
                r_name = rel['r']
                quality['negation_total'] += 1
                # Check if negation/temporal is preserved
                if any(w in r_name for w in ('not', 'stop', 'reject', 'refuse', 'former', 'previous', 'no_longer')):
                    quality['negation_correct'] += 1
                elif any(w in r_name for w in ('uses', 'prefers', 'switch')):
                    quality['negation_correct'] += 1  # neutral is ok

        if test_name == 'first_person_tech':
            quality['selfref_total'] += 1
            if any(rel['s'] == 'steven' for rel in rels):
                quality['selfref_correct'] += 1

        # Print results
        total_time = fact_time + graph_time
        marker = ''
        if is_noise and (len(facts) > 0 or len(rels) > 0):
            marker = ' !!NOISE-LEAK!!'
        print(f'\n  [{test_name}] F:{len(facts)} E:{len(emap)} R:{len(rels)} filt:{len(filtered)} '
              f'({fact_time:.1f}s+{graph_time:.1f}s={total_time:.1f}s){marker}')

        # Show facts (max 6)
        for f in facts[:6]:
            print(f'    F: {f[:100]}')
        if len(facts) > 6:
            print(f'    ... +{len(facts)-6} more facts')

        # Show relationships (max 8)
        for rel in rels[:8]:
            src_type = emap.get(rel['s'], '?')
            print(f'    R: {rel["s"]}({src_type}) --{rel["r"]}--> {rel["t"]}')
        if len(rels) > 8:
            print(f'    ... +{len(rels)-8} more rels')

        # Show filtered (max 3)
        if filtered:
            print(f'    FILTERED: {", ".join(filtered[:3])}{"..." if len(filtered)>3 else ""}')

        results[test_name] = {
            'facts': len(facts), 'ents': len(emap), 'rels': len(rels),
            'filtered': len(filtered), 'time': total_time,
            'fact_list': facts, 'rel_list': rels, 'ent_map': emap,
        }

    # Summary
    print(f'\n{"─" * 75}')
    print(f'TOTALS for {short}:')
    print(f'  Facts: {totals["facts"]}  Entities: {totals["ents"]}  Relationships: {totals["rels"]}  Filtered: {totals["filtered"]}')
    print(f'  Fact time: {totals["fact_time"]:.1f}s  Graph time: {totals["graph_time"]:.1f}s  Total: {totals["fact_time"]+totals["graph_time"]:.1f}s')
    print(f'  Avg per test: {(totals["fact_time"]+totals["graph_time"])/len(TESTS):.1f}s')
    print(f'  Tokens: facts={totals["fact_tokens"]}  graph={totals["graph_tokens"]}')
    noise_clean = quality["noise_tests"] - quality["noise_false_pos"]
    print(f'  Noise rejection: {noise_clean}/{quality["noise_tests"]} clean')
    print(f'  Self-ref (I->steven): {quality["selfref_correct"]}/{quality["selfref_total"]}')

    return results, totals, quality


if __name__ == '__main__':
    models = [
        'qwen3:4b-instruct-2507-q4_K_M',
        'qwen3.5:4b',
        'qwen3.5:9b',
    ]

    all_results = {}
    all_totals = {}
    all_quality = {}

    for model in models:
        results, totals, quality = run_model(model)
        all_results[model] = results
        all_totals[model] = totals
        all_quality[model] = quality

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print(f'\n\n{"=" * 75}')
    print('DETAILED COMPARISON')
    print(f'{"=" * 75}')

    # Per-test comparison
    header = f'{"Test":<25}'
    for m in models:
        header += f' {m.split(":")[0]:>12}'
    print(f'\n{header}')
    print('─' * len(header))

    for test_name in TESTS:
        row = f'{test_name:<25}'
        for m in models:
            r = all_results[m][test_name]
            row += f' {r["facts"]:>2}f/{r["rels"]:>2}r {r["time"]:>4.1f}s'
        print(row)

    # Totals
    print('─' * len(header))
    row = f'{"TOTAL":<25}'
    for m in models:
        t = all_totals[m]
        row += f' {t["facts"]:>2}f/{t["rels"]:>2}r {t["fact_time"]+t["graph_time"]:>4.1f}s'
    print(row)

    # Quality metrics
    print(f'\n{"Quality Metrics":<25}', end='')
    for m in models:
        print(f' {m.split(":")[0]:>12}', end='')
    print()
    print('─' * 75)

    row = f'{"Noise rejection":<25}'
    for m in models:
        q = all_quality[m]
        clean = q["noise_tests"] - q["noise_false_pos"]
        row += f' {clean}/{q["noise_tests"]:>10}'
    print(row)

    row = f'{"Self-ref (I->user)":<25}'
    for m in models:
        q = all_quality[m]
        row += f' {q["selfref_correct"]}/{q["selfref_total"]:>10}'
    print(row)

    # Speed comparison
    print(f'\n{"Speed (avg/test)":<25}', end='')
    for m in models:
        t = all_totals[m]
        avg = (t['fact_time'] + t['graph_time']) / len(TESTS)
        print(f' {avg:>11.1f}s', end='')
    print()

    print(f'{"Speed (facts only)":<25}', end='')
    for m in models:
        print(f' {all_totals[m]["fact_time"]/len(TESTS):>11.1f}s', end='')
    print()

    print(f'{"Speed (graph only)":<25}', end='')
    for m in models:
        print(f' {all_totals[m]["graph_time"]/len(TESTS):>11.1f}s', end='')
    print()
