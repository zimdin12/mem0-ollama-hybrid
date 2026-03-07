"""
Incremental fine-tuning benchmark with quality checks.

Tests fact extraction + graph extraction quality, checking actual content
(not just counts). Each test case has expected outputs to verify.

Usage: docker cp benchmark_incremental.py openmemory-mcp:/usr/src/openmemory/ && \
       MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/benchmark_incremental.py
"""
import requests
import json
import time
import os
import sys
import copy

OLLAMA_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')

# ============================================================================
# TEST CASES — each has input text + expected facts/graph checks
# ============================================================================

TESTS = {
    # --- Fact extraction quality ---
    'trivial': {
        'text': 'Steven likes pizza.',
        'expect_facts': {
            'must_contain': ['steven', 'pizza'],  # each fact must contain these
            'min_count': 1,
            'max_count': 3,
        },
        'expect_graph': {
            'must_have_rels': [('steven', 'pizza')],  # (source_contains, target_contains)
        },
    },
    'negation': {
        'text': 'Steven does not use Java. He stopped using Python last year. He rejected Ruby.',
        'expect_facts': {
            'must_contain_any': [
                ['not', 'java'],
                ['stop', 'python'],
                ['reject', 'ruby'],
            ],
            'min_count': 3,
        },
        'expect_graph': {
            'must_have_neg_rels': True,  # should have does_not_use/stopped_using/rejected
            'bad_rels': [('steven', 'uses', 'java'), ('steven', 'uses', 'python')],  # these would be WRONG
        },
    },
    'temporal': {
        'text': 'Steven previously worked at Acme Corp as a PHP developer but quit in 2024. He now works at TechStart.',
        'expect_facts': {
            'must_contain_any': [
                ['acme', 'quit'],
                ['acme', 'previous'],
                ['techstart'],
            ],
            'min_count': 3,
        },
        'expect_graph': {
            'must_have_rels': [('steven', 'techstart')],
            'temporal_check': {
                'acme': ['formerly', 'previous', 'quit', 'worked_at', 'former'],  # should be past tense
            },
        },
    },
    'multi_person': {
        'text': 'Steven is a PHP developer. Jake works with Maria on the DevOps team.',
        'expect_facts': {
            'min_count': 3,
        },
        'expect_graph': {
            'must_have_rels': [
                ('steven', 'php'),
                ('jake', 'devops'),
            ],
            'jake_devops': True,  # critical check: jake should connect to devops
            'bad_rels': [('steven', 'devops')],  # steven is NOT on devops
        },
    },
    'project_context': {
        'text': 'Echoes of the Fallen is a voxel-based roguelike built with Godot and C++. It features dual contouring and marching cubes for terrain. Development took 18 months solo.',
        'expect_facts': {
            'must_contain_any': [
                ['echoes', 'voxel'],
                ['echoes', 'roguelike'],
                ['echoes', 'godot'],
                ['echoes', 'c++'],
                ['echoes', 'dual contouring'],
                ['echoes', '18 months'],
            ],
            'context_check': True,  # every fact should mention "echoes" or the project
            'min_count': 4,
        },
        'expect_graph': {
            'must_have_rels': [
                ('echoes', 'godot'),
                ('echoes', 'c++'),
                ('echoes', 'dual_contouring'),
            ],
            'project_is_hub': 'echoes',  # echoes should be a source, not target
        },
    },
    'noise_greeting': {
        'text': 'Hey, how are you doing today?',
        'expect_facts': {'max_count': 0},
        'expect_graph': {'max_rels': 0},
        'is_noise': True,
    },
    'noise_instruction': {
        'text': 'Search for Python tutorials and show me the best ones.',
        'expect_facts': {'max_count': 0},
        'expect_graph': {'max_rels': 0},
        'is_noise': True,
    },
    'first_person': {
        'text': 'I use Neovim and prefer dark themes. My favorite language is Rust.',
        'expect_facts': {
            'must_contain_any': [
                ['neovim'],
                ['dark'],
                ['rust'],
            ],
            'no_first_person': True,  # facts should say "steven", not "I"
            'min_count': 3,
        },
        'expect_graph': {
            'must_have_rels': [
                ('steven', 'neovim'),
                ('steven', 'rust'),
            ],
            'no_i_refs': True,  # graph should use steven, not I/me
        },
    },
    'dense_technical': {
        'text': 'The project uses PostgreSQL with pgvector for embeddings, Redis for caching, and Nginx as reverse proxy. The API is built with FastAPI on Python 3.12. Docker Compose orchestrates all services. CI/CD runs on GitHub Actions with pytest.',
        'expect_facts': {
            'min_count': 5,
        },
        'expect_graph': {
            'min_rels': 4,
            'must_have_entity_types': ['technology', 'tool'],  # should have diverse types
        },
    },
    'mixed_estonian': {
        'text': 'Steven elab Tallinnas ja töötab IT-sektoris. He prefers Linux over Windows.',
        'expect_facts': {
            'must_contain_any': [
                ['tallinn'],
                ['linux'],
                ['windows'],
            ],
            'min_count': 2,
        },
        'expect_graph': {
            'must_have_rels': [('steven', 'linux')],
        },
    },
    'wrong_attribution': {
        'text': 'Steven built Echoes of the Fallen with Godot. Jake and Maria work at CloudTech on the DevOps team. Steven previously worked at Acme Corp.',
        'expect_graph': {
            'must_have_rels': [
                ('steven', 'echoes'),
                ('jake', 'cloudtech'),
                ('maria', 'cloudtech'),
            ],
            'bad_rels': [
                ('jake', 'acme'),    # jake does NOT work at acme
                ('maria', 'acme'),   # maria does NOT work at acme
                ('steven', 'devops'),  # steven is NOT on devops
            ],
        },
    },
}

# ============================================================================
# Extraction functions
# ============================================================================

def extract_facts(model, text, options):
    from custom_update_prompt import QWEN3_FACT_EXTRACTION_PROMPT as FACT_PROMPT
    resp = requests.post(f'{OLLAMA_URL}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': FACT_PROMPT},
            {'role': 'user', 'content': text},
        ],
        'stream': False, 'think': False,
        'options': {**options, 'num_predict': 2048},
    }, timeout=180)
    if resp.status_code != 200:
        return []
    content = resp.json().get('message', {}).get('content', '')
    facts = []
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
    for line in content.strip().split('\n'):
        line = line.strip().lstrip('-•* ').strip()
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
            if line in ('{', '}', '[', ']', '"facts": ['):
                continue
            if line.startswith('"') and (line.endswith('"') or line.endswith('",') or line.endswith('",')):
                line = line.strip('"').rstrip(',').strip('"').strip()
            facts.append(line)
    return facts


def extract_graph(model, text, options, user_id='steven'):
    from fix_graph_entity_parsing import GRAPH_EXTRACTION_PROMPT, _strip_json_fences
    prompt = GRAPH_EXTRACTION_PROMPT.replace('USER_ID', user_id)
    prompt += (
        "\n\nIMPORTANT: You MUST generate relationships for every entity. "
        "Every entity should connect to at least one person or project via a relationship. "
        "If an entity has no relationship, remove it from entities too."
        "\nPreserve NEGATIONS: does_not_use, stopped_using, rejected, never_used (NOT just 'uses')."
        "\nPreserve TEMPORAL state: formerly_worked_at vs currently_works_at, previously_used vs uses."
        f"\nALWAYS include a \"{user_id}\" (type: person) entity. If text says \"the project\" or "
        f"\"my project\", create \"{user_id}\" as source with relation \"uses\" to all technologies mentioned."
    )
    resp = requests.post(f'{OLLAMA_URL}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text},
        ],
        'format': 'json', 'stream': False, 'think': False,
        'options': {**options, 'num_predict': 4096},
    }, timeout=300)
    if resp.status_code != 200:
        return [], []
    content = resp.json().get('message', {}).get('content', '')
    try:
        parsed = json.loads(_strip_json_fences(content))
        entities = parsed.get('entities', [])
        rels = parsed.get('relationships', [])
        return entities, rels
    except json.JSONDecodeError:
        return [], []


def noise_check(model, text, options):
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
        return True
    try:
        result = json.loads(resp.json()['message']['content'])
        return result.get('keep', True)
    except:
        return True


# ============================================================================
# Quality scoring
# ============================================================================

def score_facts(facts, expect):
    """Score fact extraction quality. Returns (score, issues)."""
    score = 0
    issues = []

    min_c = expect.get('min_count', 0)
    max_c = expect.get('max_count', 999)

    if len(facts) < min_c:
        issues.append(f'too few facts: {len(facts)} < {min_c}')
        score -= 5
    elif len(facts) > max_c:
        issues.append(f'too many facts: {len(facts)} > {max_c}')
        score -= 10
    elif len(facts) > 100:
        issues.append(f'THINKING MODEL? {len(facts)} facts')
        score -= 30
    else:
        score += 5  # correct count range

    # Check must_contain (every fact must contain all keywords)
    if 'must_contain' in expect:
        for kw in expect['must_contain']:
            found = any(kw.lower() in f.lower() for f in facts)
            if found:
                score += 3
            else:
                issues.append(f'missing keyword in facts: {kw}')
                score -= 3

    # Check must_contain_any (at least one fact must match each keyword group)
    if 'must_contain_any' in expect:
        for kw_group in expect['must_contain_any']:
            found = False
            for f in facts:
                fl = f.lower()
                if all(kw.lower() in fl for kw in kw_group):
                    found = True
                    break
            if found:
                score += 3
            else:
                issues.append(f'no fact matches: {kw_group}')
                score -= 3

    # Check context_check: every fact should reference the project
    if expect.get('context_check'):
        orphaned = 0
        for f in facts:
            fl = f.lower()
            if 'echoes' not in fl and 'fallen' not in fl and 'game' not in fl and 'project' not in fl:
                orphaned += 1
        if orphaned > 0:
            issues.append(f'{orphaned}/{len(facts)} facts missing project context')
            score -= orphaned * 2

    # Check no_first_person: facts should use "steven" not "I"
    if expect.get('no_first_person') and facts:
        i_count = sum(1 for f in facts if f.strip().startswith('I ') or ' I ' in f)
        steven_count = sum(1 for f in facts if 'steven' in f.lower())
        if i_count > 0 and steven_count == 0:
            issues.append(f'{i_count} facts use "I" instead of "Steven"')
            score -= 5
        elif steven_count > 0:
            score += 3

    return score, issues


def score_graph(entities, rels, expect):
    """Score graph extraction quality. Returns (score, issues)."""
    score = 0
    issues = []

    # Normalize for checking
    def norm(s):
        return str(s).lower().replace(' ', '_')

    rel_tuples = []
    for r in rels:
        src = norm(r.get('source', ''))
        tgt = norm(r.get('target', ''))
        rel_name = norm(r.get('relation', r.get('relationship', '')))
        rel_tuples.append((src, rel_name, tgt))

    # Check must_have_rels: (source_contains, target_contains)
    if 'must_have_rels' in expect:
        for src_kw, tgt_kw in expect['must_have_rels']:
            found = any(src_kw in s and tgt_kw in t for s, _, t in rel_tuples)
            if found:
                score += 5
            else:
                issues.append(f'missing rel: {src_kw} -> {tgt_kw}')
                score -= 5

    # Check bad_rels: these should NOT exist
    if 'bad_rels' in expect:
        for bad in expect['bad_rels']:
            if len(bad) == 2:
                src_kw, tgt_kw = bad
                found = any(src_kw in s and tgt_kw in t for s, _, t in rel_tuples)
            else:
                src_kw, rel_kw, tgt_kw = bad
                found = any(src_kw in s and rel_kw in r and tgt_kw in t for s, r, t in rel_tuples)
            if found:
                issues.append(f'BAD rel present: {bad}')
                score -= 8

    # Check jake_devops
    if expect.get('jake_devops'):
        found = any('jake' in s and 'devops' in t for s, _, t in rel_tuples)
        if found:
            score += 10
        else:
            issues.append('jake NOT connected to devops')
            score -= 5

    # Check negation relationships
    if expect.get('must_have_neg_rels'):
        neg_rels = [r for _, r, _ in rel_tuples if any(neg in r for neg in
                    ['not', 'stop', 'reject', 'never', 'former', 'quit', 'left'])]
        if neg_rels:
            score += 5 * len(neg_rels)
        else:
            issues.append('no negation relationships found')
            score -= 5

    # Check temporal
    if 'temporal_check' in expect:
        for entity_kw, expected_terms in expect['temporal_check'].items():
            relevant = [(s, r, t) for s, r, t in rel_tuples if entity_kw in t or entity_kw in s]
            if relevant:
                has_temporal = any(any(term in r for term in expected_terms) for _, r, _ in relevant)
                if has_temporal:
                    score += 5
                else:
                    rel_names = [r for _, r, _ in relevant]
                    issues.append(f'temporal not preserved for {entity_kw}: got {rel_names}')
                    score -= 3

    # Check project_is_hub
    if 'project_is_hub' in expect:
        proj_kw = expect['project_is_hub']
        as_source = sum(1 for s, _, _ in rel_tuples if proj_kw in s)
        if as_source >= 2:
            score += 5
        else:
            issues.append(f'{proj_kw} not acting as hub (only {as_source} outgoing)')
            score -= 3

    # Check no_i_refs
    if expect.get('no_i_refs'):
        i_refs = any('i' == s or 'me' == s or 'my' == s for s, _, _ in rel_tuples)
        if i_refs:
            issues.append('graph has I/me refs instead of steven')
            score -= 5
        else:
            score += 3

    # Check max_rels
    max_r = expect.get('max_rels')
    if max_r is not None and len(rels) > max_r:
        issues.append(f'too many rels for noise: {len(rels)}')
        score -= 5

    # Check min_rels
    min_r = expect.get('min_rels')
    if min_r is not None and len(rels) < min_r:
        issues.append(f'too few rels: {len(rels)} < {min_r}')
        score -= 3

    # Entity type diversity
    if 'must_have_entity_types' in expect:
        etypes = set()
        for e in entities:
            if isinstance(e, dict):
                etypes.add(e.get('type', '').lower())
        for et in expect['must_have_entity_types']:
            if et not in etypes:
                issues.append(f'missing entity type: {et}')
                score -= 2

    # Self-ref check (always)
    for s, r, t in rel_tuples:
        if s == t:
            issues.append(f'SELF-REF: {s} --{r}--> {t}')
            score -= 10

    return score, issues


# ============================================================================
# Run test
# ============================================================================

def run_single_test(model, test_name, test_data, options):
    """Run one test case. Returns (fact_score, graph_score, fact_issues, graph_issues, details)."""
    text = test_data['text']
    details = {}

    # Noise check for short texts
    if test_data.get('is_noise') or len(text.strip()) < 100:
        keep = noise_check(model, text, options)
        if test_data.get('is_noise'):
            if not keep:
                return 10, 5, [], [], {'noise': 'correctly dropped'}
            else:
                return -10, -5, ['noise leaked'], ['noise leaked'], {'noise': 'LEAKED'}

    # Fact extraction
    t0 = time.time()
    facts = extract_facts(model, text, options)
    fact_time = time.time() - t0

    # Graph extraction
    t0 = time.time()
    entities, rels = extract_graph(model, text, options)
    graph_time = time.time() - t0

    details['facts'] = facts
    details['fact_count'] = len(facts)
    details['rel_count'] = len(rels)
    details['fact_time'] = round(fact_time, 1)
    details['graph_time'] = round(graph_time, 1)
    details['entities'] = entities
    details['rels'] = rels

    fact_score, fact_issues = score_facts(facts, test_data.get('expect_facts', {}))
    graph_score, graph_issues = score_graph(entities, rels, test_data.get('expect_graph', {}))

    return fact_score, graph_score, fact_issues, graph_issues, details


def run_all_tests(model, options, label='', verbose=True):
    """Run all tests and return total score + breakdown."""
    total_fact = 0
    total_graph = 0
    all_issues = []
    total_facts = 0
    total_rels = 0

    for test_name, test_data in TESTS.items():
        fs, gs, fi, gi, det = run_single_test(model, test_name, test_data, options)
        total_fact += fs
        total_graph += gs
        total_facts += det.get('fact_count', 0)
        total_rels += det.get('rel_count', 0)

        status_parts = []
        if det.get('noise'):
            status_parts.append(det['noise'])
        else:
            status_parts.append(f"{det.get('fact_count',0)}f/{det.get('rel_count',0)}r")
            status_parts.append(f"{det.get('fact_time',0)}+{det.get('graph_time',0)}s")

        issues = fi + gi
        if issues:
            status_parts.append(f"ISSUES: {'; '.join(issues)}")
            all_issues.extend([(test_name, i) for i in issues])

        if verbose:
            print(f"    [{test_name}] f={fs:+d} g={gs:+d} | {' | '.join(status_parts)}", flush=True)

    total = total_fact + total_graph
    if verbose:
        print(f"  => TOTAL: {total} (fact={total_fact}, graph={total_graph}) | {total_facts}f/{total_rels}r | {len(all_issues)} issues", flush=True)

    return {
        'total': total,
        'fact_score': total_fact,
        'graph_score': total_graph,
        'total_facts': total_facts,
        'total_rels': total_rels,
        'issues': all_issues,
        'label': label,
    }


# ============================================================================
# Incremental tuning configs
# ============================================================================

MODELS = {
    'ministral-3:3b': {
        'baseline': {'temperature': 0.05, 'top_p': 0.8, 'top_k': 40},
        'tweaks': [
            ('t0.03', {'temperature': 0.03}),
            ('t0.08', {'temperature': 0.08}),
            ('t0.01', {'temperature': 0.01}),
            ('k50', {'top_k': 50}),
            ('k30', {'top_k': 30}),
            ('k20', {'top_k': 20}),
            ('p0.9', {'top_p': 0.9}),
            ('p0.7', {'top_p': 0.7}),
        ],
    },
    'qwen3.5:4b': {
        'baseline': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5},
        'tweaks': [
            ('t0.1', {'temperature': 0.1}),
            ('t0.2', {'temperature': 0.2}),
            ('t0.12', {'temperature': 0.12}),
            ('t0.18', {'temperature': 0.18}),
            ('pp0.3', {'presence_penalty': 0.3}),
            ('pp0.7', {'presence_penalty': 0.7}),
            ('pp1.0', {'presence_penalty': 1.0}),
            ('k30', {'top_k': 30}),
            ('k40', {'top_k': 40}),
        ],
    },
    'qwen3.5:9b': {
        'baseline': {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5},
        'tweaks': [
            ('t0.08', {'temperature': 0.08}),
            ('t0.12', {'temperature': 0.12}),
            ('t0.15', {'temperature': 0.15}),
            ('pp1.0', {'presence_penalty': 1.0}),
            ('pp1.2', {'presence_penalty': 1.2}),
            ('pp1.8', {'presence_penalty': 1.8}),
            ('k30', {'top_k': 30}),
            ('k40', {'top_k': 40}),
        ],
    },
    'qwen3:4b-instruct-2507-q4_K_M': {
        'baseline': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20},
        'tweaks': [
            ('t0.5', {'temperature': 0.5}),
            ('t0.6', {'temperature': 0.6}),
            ('t0.8', {'temperature': 0.8}),
            ('t0.9', {'temperature': 0.9}),
            ('k30', {'top_k': 30}),
            ('k40', {'top_k': 40}),
            ('p0.9', {'top_p': 0.9}),
            ('p0.7', {'top_p': 0.7}),
        ],
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Test only this model')
    parser.add_argument('--baseline-only', action='store_true', help='Only test baselines')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-test details')
    args = parser.parse_args()

    models_to_test = MODELS
    if args.model:
        models_to_test = {k: v for k, v in MODELS.items() if args.model in k}
        if not models_to_test:
            print(f"No model matching '{args.model}'. Available: {list(MODELS.keys())}")
            sys.exit(1)

    print("=" * 80, flush=True)
    print("INCREMENTAL FINE-TUNING BENCHMARK (quality-focused)", flush=True)
    print("=" * 80, flush=True)

    for model, config in models_to_test.items():
        print(f"\n{'='*80}", flush=True)
        print(f"MODEL: {model}", flush=True)
        print(f"{'='*80}", flush=True)

        # Warmup
        print("  Warming up...", end=' ', flush=True)
        try:
            requests.post(f'{OLLAMA_URL}/api/chat', json={
                'model': model,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'stream': False, 'think': False,
                'options': {'num_predict': 1},
            }, timeout=120)
            print("ready", flush=True)
        except:
            print("TIMEOUT - skipping", flush=True)
            continue

        # Baseline
        baseline = config['baseline']
        print(f"\n  --- BASELINE: {baseline} ---", flush=True)
        baseline_result = run_all_tests(model, baseline, 'baseline', verbose=True)

        if args.baseline_only:
            continue

        # Incremental tweaks — one param change at a time
        results = [baseline_result]
        for tweak_name, tweak_overrides in config['tweaks']:
            opts = {**baseline, **tweak_overrides}
            changed = {k: v for k, v in tweak_overrides.items() if baseline.get(k) != v}
            print(f"\n  --- {tweak_name}: {changed} ---", flush=True)
            result = run_all_tests(model, opts, tweak_name, verbose=True)
            results.append(result)

        # Summary ranking
        results.sort(key=lambda r: r['total'], reverse=True)
        print(f"\n  RANKING for {model}:", flush=True)
        for i, r in enumerate(results):
            marker = ' <<<< BEST' if i == 0 else (' <<<< WORST' if i == len(results) - 1 else '')
            issue_count = len(r['issues'])
            print(f"    {i+1}. {r['label']}: score={r['total']} (f={r['fact_score']},g={r['graph_score']}) "
                  f"| {r['total_facts']}f/{r['total_rels']}r | {issue_count} issues{marker}", flush=True)

        # Show issues for best config
        best = results[0]
        if best['issues']:
            print(f"\n  Issues in best config ({best['label']}):", flush=True)
            for test_name, issue in best['issues']:
                print(f"    [{test_name}] {issue}", flush=True)


if __name__ == '__main__':
    main()
