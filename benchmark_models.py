"""
Cross-model benchmark for memory extraction pipeline.
Tests each model through the actual Ollama API with production prompts+filters.
Clears Ollama context between models to avoid contamination.

Usage: docker cp benchmark_models.py openmemory-mcp:/usr/src/openmemory/ && \
       docker exec openmemory-mcp python3 benchmark_models.py
"""
import requests
import json
import time
import os
import sys

OLLAMA_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
EMBEDDER = os.environ.get('EMBEDDER_MODEL', 'qwen3-embedding:0.6b')

MODELS = [
    'qwen3:4b-instruct-2507-q4_K_M',
    'qwen3.5:9b',
    'ministral-3:3b',
    'granite4:3b',
    'gemma3:4b',
    'phi4-mini-reasoning',
    'lfm2.5-thinking',
    'exaone-deep:2.4b',
    'deepscaler',
]

# --- Test cases ---
TESTS = {
    'trivial': {
        'text': 'Steven likes pizza.',
        'expect_facts': 1, 'expect_noise': False,
    },
    'short_basic': {
        'text': 'Steven is 28 years old. He lives in Tallinn and works as a game developer.',
        'expect_facts': 3, 'expect_noise': False,
    },
    'noise_greeting': {
        'text': 'Hey, how are you? Just checking in. Nothing much going on today.',
        'expect_facts': 0, 'expect_noise': True,
    },
    'noise_estonian': {
        'text': 'Tere, kuidas läheb? Täna on ilus ilm, midagi erilist ei toimu.',
        'expect_facts': 0, 'expect_noise': True,
    },
    'noise_instruction': {
        'text': 'Search my memories about Python and show me what you know.',
        'expect_facts': 0, 'expect_noise': True,
    },
    'negation': {
        'text': 'Steven does NOT use Docker Desktop. He runs Docker Engine directly on Linux. He tried Podman but went back to Docker because Podman had compatibility issues. He refuses to use Snap packages.',
        'expect_facts': 4, 'expect_noise': False,
        'check_negation': True,
    },
    'first_person': {
        'text': 'I have been using Rust for about 6 months. I switched from C++ because I was tired of memory bugs. My main project is a voxel engine called CrystalForge.',
        'expect_facts': 3, 'expect_noise': False,
        'check_self_ref': True,
    },
    'multi_person': {
        'text': 'Steven leads the Echoes of the Fallen project. Maria does 3D art using Blender. Jake handles DevOps. Maria reports to Steven, and Jake reports to Maria.',
        'expect_facts': 4, 'expect_noise': False,
        'check_multi_person': True,
    },
    'temporal': {
        'text': 'Steven started coding in PHP in 2015. He moved to Python in 2018 for machine learning. In 2022 he picked up Rust which is now his primary language. He still uses Python for scripting but no longer writes PHP at all.',
        'expect_facts': 4, 'expect_noise': False,
    },
    'dense_technical': {
        'text': 'The CI/CD pipeline uses GitHub Actions with 3 workflows. The build workflow runs on ubuntu-latest with Rust 1.78. The test workflow runs in parallel across Windows, Linux, and macOS using coverage via tarpaulin. The deploy workflow pushes Docker images to ghcr.io tagged with git SHA and triggers ArgoCD sync to the k8s cluster. Build caching uses sccache with an S3 backend on Hetzner.',
        'expect_facts': 6, 'expect_noise': False,
    },
    'mixed_estonian': {
        'text': "Steven's project documentation is in English but his personal notes are in Estonian. He calls his test database 'proov_andmebaas' which means test database in Estonian. His git branch naming uses English with feature/, bugfix/, hotfix/ prefixes.",
        'expect_facts': 4, 'expect_noise': False,
    },
}

def unload_model(model):
    """Unload a model from Ollama to clear its context."""
    try:
        requests.post(f'{OLLAMA_URL}/api/generate', json={
            'model': model, 'keep_alive': 0
        }, timeout=10)
    except:
        pass

def warm_up(model):
    """Send a short request to ensure model is loaded."""
    try:
        requests.post(f'{OLLAMA_URL}/api/chat', json={
            'model': model,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'stream': False,
            'options': {'num_predict': 1},
        }, timeout=30)
    except:
        pass

def extract_facts(model, text, options):
    """Call the production fact extraction prompt."""
    from custom_update_prompt import QWEN3_FACT_EXTRACTION_PROMPT as FACT_PROMPT

    resp = requests.post(f'{OLLAMA_URL}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': FACT_PROMPT},
            {'role': 'user', 'content': text},
        ],
        'stream': False,
        'think': False,
        'options': {**options, 'num_predict': 2048},
    }, timeout=180)
    if resp.status_code != 200:
        return []
    content = resp.json().get('message', {}).get('content', '')
    # Parse facts — handle both JSON and line-by-line output
    facts = []
    # Try JSON parsing first (handles single-line JSON like {"facts": ["a", "b"]})
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict) and 'facts' in parsed:
            for f in parsed['facts']:
                f = str(f).strip()
                if f and len(f) >= 10:
                    facts.append(f)
            return facts
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: line-by-line splitting
    for line in content.strip().split('\n'):
        line = line.strip().lstrip('-•* ').strip()
        # Try to extract JSON from individual lines
        if line.startswith('{'):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict) and 'facts' in parsed:
                    for f in parsed['facts']:
                        f = str(f).strip()
                        if f and len(f) >= 10:
                            facts.append(f)
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
        if line and len(line) >= 10:
            # Skip JSON structure lines
            if line in ('{', '}', '[', ']', '"facts": ['):
                continue
            # Clean quoted fact strings
            if line.startswith('"') and (line.endswith('"') or line.endswith('",') or line.endswith('",')):
                line = line.strip('"').rstrip(',').strip('"').strip()
            facts.append(line)
    return facts

def extract_graph(model, text, options):
    """Call the production graph extraction prompt."""
    from fix_graph_entity_parsing import (
        GRAPH_EXTRACTION_PROMPT, _strip_json_fences, _detect_model_family
    )

    prompt = GRAPH_EXTRACTION_PROMPT.replace('USER_ID', 'steven')
    # Add universal suffix
    prompt += (
        "\n\nIMPORTANT: You MUST generate relationships for every entity. "
        "Every entity should connect to at least one person or project via a relationship. "
        "If an entity has no relationship, remove it from entities too."
        "\nPreserve NEGATIONS: does_not_use, stopped_using, rejected, never_used (NOT just 'uses')."
        "\nPreserve TEMPORAL state: formerly_worked_at vs currently_works_at, previously_used vs uses."
    )

    resp = requests.post(f'{OLLAMA_URL}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text},
        ],
        'format': 'json',
        'stream': False,
        'think': False,
        'options': {**options, 'num_predict': 4096},
    }, timeout=300)
    if resp.status_code != 200:
        return {}, []
    content = resp.json().get('message', {}).get('content', '')
    try:
        parsed = json.loads(_strip_json_fences(content))
        entities = parsed.get('entities', [])
        rels = parsed.get('relationships', [])
        return entities, rels
    except json.JSONDecodeError:
        return [], []

def noise_check(model, text, options):
    """Call the production noise classification."""
    resp = requests.post(f'{OLLAMA_URL}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You classify text for a memory system. Respond with JSON only.'},
            {'role': 'user', 'content': (
                'Does this text state a FACT about someone or something? '
                'KEEP only if it declares a preference, skill, decision, project detail, or personal info. '
                'DROP if it is: a greeting, small talk, weather, daily routine, '
                'a question, a request/command (even if it mentions technologies like "search Python" '
                'or "show me Rust"), or an acknowledgment ("ok", "thanks", "got it"). '
                'The key test: does it TELL something new, or does it ASK/REQUEST something?\n\n'
                f'Text: "{text}"\n\nRespond: {{"keep": true/false}}'
            )},
        ],
        'stream': False, 'format': 'json', 'think': False,
        'options': {**options, 'num_predict': 32},
    }, timeout=15)
    if resp.status_code != 200:
        return True  # default: keep
    try:
        result = json.loads(resp.json()['message']['content'])
        return result.get('keep', True)
    except:
        return True

def run_test(model, test_name, test_data, options):
    """Run a single test and return results."""
    text = test_data['text']
    result = {'name': test_name}

    # Noise check (for short texts)
    if len(text.strip()) < 100:
        t0 = time.time()
        keep = noise_check(model, text, options)
        result['noise_time'] = time.time() - t0
        result['noise_kept'] = keep
        if not keep:
            result['facts'] = []
            result['entities'] = []
            result['relationships'] = []
            result['fact_time'] = 0
            result['graph_time'] = 0
            return result

    # Fact extraction
    t0 = time.time()
    facts = extract_facts(model, text, options)
    result['fact_time'] = time.time() - t0
    result['facts'] = facts

    # Graph extraction
    t0 = time.time()
    entities, rels = extract_graph(model, text, options)
    result['graph_time'] = time.time() - t0
    result['entities'] = entities
    result['relationships'] = rels

    return result


def main():
    from model_configs import detect_model_family as _detect_model_family, get_llm_options as _get_llm_options

    # Parse CLI args
    override_options = None
    skip_thinking = False
    for arg in sys.argv[1:]:
        if arg == '--ministral-config':
            override_options = {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20}
            print(f"OVERRIDE: Using ministral config for ALL models: {override_options}")
        elif arg == '--skip-thinking':
            skip_thinking = True
            print("SKIP: Excluding thinking models (phi4, lfm2.5, exaone-deep, deepscaler)")

    THINKING_MODELS = {'phi4-mini-reasoning', 'lfm2.5-thinking', 'exaone-deep:2.4b', 'deepscaler'}

    # Check which models are available
    available = []
    for model in MODELS:
        if skip_thinking and model in THINKING_MODELS:
            print(f"  Skipped: {model} (thinking model)")
            continue
        try:
            r = requests.post(f'{OLLAMA_URL}/api/show', json={'model': model}, timeout=5)
            if r.status_code == 200:
                available.append(model)
                print(f"  Found: {model}")
            else:
                print(f"  Missing: {model}")
        except:
            print(f"  Error checking: {model}")

    if not available:
        print("No models available!")
        sys.exit(1)

    all_results = {}

    for model in available:
        # Unload previous model
        for m in available:
            if m != model:
                unload_model(m)

        family = _detect_model_family(model)
        options = override_options if override_options else _get_llm_options(family)

        print(f"\n{'='*75}")
        print(f"MODEL: {model} (family={family}){' [MINISTRAL CONFIG]' if override_options else ''}")
        print(f"Options: {options}")
        print(f"{'='*75}")

        # Warm up
        print("  Warming up...", end=' ', flush=True)
        warm_up(model)
        print("ready")

        model_results = {}
        total_fact_time = 0
        total_graph_time = 0

        for test_name, test_data in TESTS.items():
            result = run_test(model, test_name, test_data, options)
            model_results[test_name] = result

            ft = result.get('fact_time', 0)
            gt = result.get('graph_time', 0)
            nt = result.get('noise_time', 0)
            total_fact_time += ft
            total_graph_time += gt

            facts = result.get('facts', [])
            ents = result.get('entities', [])
            rels = result.get('relationships', [])

            total_t = ft + gt + nt
            noise_status = ""
            if 'noise_kept' in result:
                noise_status = " KEPT" if result['noise_kept'] else " DROPPED"
                if test_data.get('expect_noise') and result['noise_kept']:
                    noise_status += " !!LEAK!!"
                elif not test_data.get('expect_noise') and not result['noise_kept']:
                    noise_status += " !!LOST!!"

            print(f"\n  [{test_name}] F:{len(facts)} E:{len(ents)} R:{len(rels)} ({total_t:.1f}s){noise_status}")

            # Print facts
            for f in facts[:4]:
                print(f"    F: {f[:90]}")
            if len(facts) > 4:
                print(f"    ... +{len(facts)-4} more")

            # Print relationships
            for r in rels[:5]:
                if isinstance(r, dict):
                    src = r.get('source', '?')
                    rel = r.get('relation', '?')
                    tgt = r.get('target', '?')
                    print(f"    R: {src} --{rel}--> {tgt}")
            if len(rels) > 5:
                print(f"    ... +{len(rels)-5} more rels")

            # Quality checks
            if test_data.get('check_negation'):
                neg_rels = [r for r in rels if isinstance(r, dict) and
                           any(neg in r.get('relation', '') for neg in
                               ['not', 'stop', 'refus', 'reject', 'never', 'former'])]
                print(f"    Negation rels: {len(neg_rels)}")

            if test_data.get('check_self_ref'):
                self_refs = [r for r in rels if isinstance(r, dict) and
                            r.get('source', '').lower() in ('steven', 'user')]
                i_refs = [r for r in rels if isinstance(r, dict) and
                         r.get('source', '').lower() in ('i', 'me', 'my', 'myself')]
                print(f"    I->steven: {len(self_refs)} correct, {len(i_refs)} unreplaced")

            if test_data.get('check_multi_person'):
                jake_rels = [r for r in rels if isinstance(r, dict) and
                            r.get('source', '').lower() == 'jake']
                jake_devops = any(r for r in rels if isinstance(r, dict) and
                                 r.get('source', '').lower() == 'jake' and
                                 'devops' in r.get('target', '').lower())
                maria_reports = any(r for r in rels if isinstance(r, dict) and
                                   r.get('source', '').lower() == 'maria' and
                                   'reports' in r.get('relation', '').lower() and
                                   r.get('target', '').lower() == 'steven')
                print(f"    Jake rels: {len(jake_rels)}, jake->devops: {jake_devops}, maria->steven: {maria_reports}")

        all_results[model] = model_results

        # Model totals
        total_facts = sum(len(r.get('facts', [])) for r in model_results.values())
        total_rels = sum(len(r.get('relationships', [])) for r in model_results.values())
        noise_tests = [(k, v) for k, v in model_results.items() if 'noise_kept' in v]
        noise_correct = sum(1 for k, v in noise_tests
                           if (TESTS[k].get('expect_noise') and not v['noise_kept']) or
                              (not TESTS[k].get('expect_noise') and v['noise_kept']))

        print(f"\n  TOTALS: {total_facts}f {total_rels}r | "
              f"Fact:{total_fact_time:.1f}s Graph:{total_graph_time:.1f}s | "
              f"Noise: {noise_correct}/{len(noise_tests)} correct")

    # --- Comparison table ---
    print(f"\n\n{'='*75}")
    print("COMPARISON TABLE")
    print(f"{'='*75}")

    # Header
    model_names = [m if ':' not in m else m.split(':')[0] + ':' + m.split(':')[1][:4] for m in available]
    header = f"{'Test':<20}" + "".join(f"{n:>16}" for n in model_names)
    print(header)
    print("-" * len(header))

    for test_name in TESTS:
        row = f"{test_name:<20}"
        for model in available:
            r = all_results.get(model, {}).get(test_name, {})
            nf = len(r.get('facts', []))
            nr = len(r.get('relationships', []))
            ft = r.get('fact_time', 0) + r.get('graph_time', 0) + r.get('noise_time', 0)
            noise = ""
            if 'noise_kept' in r and not r['noise_kept']:
                noise = "D"
            elif 'noise_kept' in r and r['noise_kept'] and TESTS[test_name].get('expect_noise'):
                noise = "!"
            row += f"{nf:>3}f/{nr:<2}r {ft:>4.1f}s{noise:>1} "
        print(row)

    # Totals row
    print("-" * len(header))
    row = f"{'TOTAL':<20}"
    for model in available:
        mr = all_results.get(model, {})
        tf = sum(len(r.get('facts', [])) for r in mr.values())
        tr = sum(len(r.get('relationships', [])) for r in mr.values())
        tt = sum(r.get('fact_time', 0) + r.get('graph_time', 0) + r.get('noise_time', 0) for r in mr.values())
        row += f"{tf:>3}f/{tr:<2}r {tt:>4.1f}s  "
    print(row)

    # Quality metrics
    print()
    row = f"{'Noise correct':<20}"
    for model in available:
        mr = all_results.get(model, {})
        noise_tests = [(k, v) for k, v in mr.items() if 'noise_kept' in v]
        correct = sum(1 for k, v in noise_tests
                     if (TESTS[k].get('expect_noise') and not v['noise_kept']) or
                        (not TESTS[k].get('expect_noise') and v['noise_kept']))
        row += f"{correct}/{len(noise_tests):>14} "
    print(row)

    row = f"{'Multi-person':<20}"
    for model in available:
        r = all_results.get(model, {}).get('multi_person', {})
        rels = r.get('relationships', [])
        jake_devops = any(r2 for r2 in rels if isinstance(r2, dict) and
                         r2.get('source', '').lower() == 'jake' and
                         'devops' in r2.get('target', '').lower())
        row += f"{'correct' if jake_devops else 'WRONG':>16}"
    print(row)


if __name__ == '__main__':
    main()
