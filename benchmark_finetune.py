"""
Config fine-tuning benchmark. Tests parameter variations on specific models.
Runs each model with multiple config combos to find optimal settings.

Usage: docker cp benchmark_finetune.py openmemory-mcp:/usr/src/openmemory/ && \
       MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/benchmark_finetune.py
"""
import requests
import json
import time
import os
import sys

OLLAMA_URL = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')

# Models to tune
# Incremental tuning: start from current config, change ONE param at a time
# to see direction of improvement, then combine winners.
# Full 11-test suite for each config to ensure valid comparisons.

MODELS_TO_TUNE = {
    'qwen3.5:4b': [
        # Baseline (old config)
        {'label': 'old_t01_pp15',       'options': {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}},
        # Remove PP
        {'label': 'noPP_t01',           'options': {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20}},
        # Lower PP values
        {'label': 'pp05_t01',           'options': {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
        # Temp variations without PP
        {'label': 'noPP_t015',          'options': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20}},
        {'label': 'noPP_t02',           'options': {'temperature': 0.2, 'top_p': 0.8, 'top_k': 20}},
        {'label': 'noPP_t03',           'options': {'temperature': 0.3, 'top_p': 0.8, 'top_k': 20}},
        # Temp variations with low PP
        {'label': 'pp05_t015',          'options': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
        {'label': 'pp05_t03',           'options': {'temperature': 0.3, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
    ],
    'hf.co/crownelius/Crow-4B-Opus-4.6-Distill-Heretic_Qwen3.5:Q4_K_M': [
        # Test with qwen3.5 configs — distilled model might need different tuning
        {'label': 'q35_t03_pp05',       'options': {'temperature': 0.3, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
        {'label': 'low_t005',           'options': {'temperature': 0.05, 'top_p': 0.8, 'top_k': 20}},
        {'label': 'low_t01',            'options': {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20}},
        {'label': 'low_t015',           'options': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20}},
        {'label': 'noPP_t03',           'options': {'temperature': 0.3, 'top_p': 0.8, 'top_k': 20}},
    ],
    'qwen3.5:9b': [
        # Baseline (old config)
        {'label': 'old_t01_pp15',       'options': {'temperature': 0.1, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5}},
        # PP variations at t=0.15
        {'label': 'pp05_t015',          'options': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
        {'label': 'noPP_t015',          'options': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20}},
        # Temp variations with pp=0.5
        {'label': 'pp05_t02',           'options': {'temperature': 0.2, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
        {'label': 'pp05_t03',           'options': {'temperature': 0.3, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 0.5}},
    ],
    'ministral-3:3b': [
        # Previous winner
        {'label': 'BEST_t005',          'options': {'temperature': 0.05, 'top_p': 0.8, 'top_k': 20}},
        # Baseline
        {'label': 'base_t015',          'options': {'temperature': 0.15, 'top_p': 0.8, 'top_k': 20}},
        # Even lower temp
        {'label': 'ultra_t001',         'options': {'temperature': 0.01, 'top_p': 0.8, 'top_k': 20}},
        # top_k at best temp
        {'label': 't005_k10',           'options': {'temperature': 0.05, 'top_p': 0.8, 'top_k': 10}},
        {'label': 't005_k40',           'options': {'temperature': 0.05, 'top_p': 0.8, 'top_k': 40}},
        # top_p at best temp
        {'label': 't005_p07',           'options': {'temperature': 0.05, 'top_p': 0.7, 'top_k': 20}},
        {'label': 't005_p09',           'options': {'temperature': 0.05, 'top_p': 0.9, 'top_k': 20}},
    ],
}

# Full test set — same as benchmark_models.py for consistent comparison
TESTS = {
    'trivial': {
        'text': 'Steven likes pizza.',
        'expect_noise': False,
    },
    'short_basic': {
        'text': 'Steven is 28 years old. He lives in Tallinn and works as a game developer.',
        'expect_noise': False,
    },
    'noise_greeting': {
        'text': 'Hey, how are you? Just checking in. Nothing special today.',
        'expect_noise': True,
    },
    'noise_estonian': {
        'text': 'Tere, kuidas läheb? Täna on ilus ilm, midagi erilist ei ole.',
        'expect_noise': True,
    },
    'noise_instruction': {
        'text': 'Search my memories about Python and list everything you know.',
        'expect_noise': True,
    },
    'negation': {
        'text': 'Steven does NOT use Docker Desktop. He runs Docker Engine directly on Linux. He tried Podman but had compatibility issues so went back to Docker. He also refuses to use Snap packages.',
    },
    'first_person': {
        'text': 'I have been using Rust for about 6 months. I switched from C++ because I was tired of memory bugs. My main project CrystalForge is a voxel engine.',
    },
    'multi_person': {
        'text': 'Steven leads the Echoes of the Fallen project. Maria does 3D art using Blender. Jake handles DevOps. Maria reports to Steven, and Jake reports to Maria.',
    },
    'temporal': {
        'text': 'Steven started coding in PHP in 2015. He moved to Python in 2018 for machine learning. In 2022 he picked up Rust which is now his primary language. He still uses Python for scripting.',
    },
    'dense_technical': {
        'text': 'The CI/CD pipeline uses GitHub Actions with 3 workflows. The build workflow runs on ubuntu-latest with Rust 1.78. The test workflow runs across Windows, Linux, and macOS using coverage via tarpaulin. Build caching uses sccache for 60% faster rebuilds. The deploy workflow builds Docker images and pushes to ghcr.io.',
    },
    'mixed_estonian': {
        'text': "Steven's project documentation is in English but his personal notes are in Estonian. He calls his test database 'proov_andmebaas'. Git branch naming uses feature/, bugfix/, hotfix/ prefixes.",
    },
}


def warm_up(model):
    try:
        requests.post(f'{OLLAMA_URL}/api/chat', json={
            'model': model,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'stream': False,
            'options': {'num_predict': 1},
        }, timeout=60)
    except:
        pass


def unload_model(model):
    try:
        requests.post(f'{OLLAMA_URL}/api/generate', json={
            'model': model, 'keep_alive': 0
        }, timeout=10)
    except:
        pass


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
    # Parse facts — handle both JSON and line-by-line output
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


def extract_graph(model, text, options):
    from fix_graph_entity_parsing import GRAPH_EXTRACTION_PROMPT, _strip_json_fences
    prompt = GRAPH_EXTRACTION_PROMPT.replace('USER_ID', 'steven')
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
        'format': 'json', 'stream': False, 'think': False,
        'options': {**options, 'num_predict': 4096},
    }, timeout=300)
    if resp.status_code != 200:
        return [], []
    content = resp.json().get('message', {}).get('content', '')
    try:
        parsed = json.loads(_strip_json_fences(content))
        return parsed.get('entities', []), parsed.get('relationships', [])
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


def run_test(model, test_name, test_data, options):
    text = test_data['text']
    result = {'name': test_name}

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

    t0 = time.time()
    facts = extract_facts(model, text, options)
    result['fact_time'] = time.time() - t0
    result['facts'] = facts

    t0 = time.time()
    entities, rels = extract_graph(model, text, options)
    result['graph_time'] = time.time() - t0
    result['entities'] = entities
    result['relationships'] = rels

    return result


def score_result(all_test_results):
    """Score a config run. Higher = better."""
    score = 0
    total_facts = 0
    total_rels = 0
    total_time = 0
    noise_correct = 0
    noise_total = 0

    for test_name, r in all_test_results.items():
        nf = len(r.get('facts', []))
        nr = len(r.get('relationships', []))
        t = r.get('fact_time', 0) + r.get('graph_time', 0) + r.get('noise_time', 0)
        total_facts += nf
        total_rels += nr
        total_time += t

        # Noise scoring
        if 'noise_kept' in r:
            noise_total += 1
            expect_noise = TESTS[test_name].get('expect_noise', False)
            if expect_noise and not r['noise_kept']:
                noise_correct += 1
                score += 10  # correct noise rejection
            elif not expect_noise and r['noise_kept']:
                noise_correct += 1
                score += 5  # correct keep
            elif expect_noise and r['noise_kept']:
                score -= 15  # noise leak is bad
            elif not expect_noise and not r['noise_kept']:
                score -= 20  # lost real data is very bad

        # Multi-person: check jake->devops
        if test_name == 'multi_person':
            rels = r.get('relationships', [])
            jake_devops = any(rel for rel in rels if isinstance(rel, dict) and
                            'jake' in rel.get('source', '').lower() and
                            'devops' in rel.get('target', '').lower())
            maria_steven = any(rel for rel in rels if isinstance(rel, dict) and
                             'maria' in rel.get('source', '').lower() and
                             'steven' in rel.get('target', '').lower())
            if jake_devops:
                score += 10
            if maria_steven:
                score += 5

        # Negation: check for negation relationships
        if test_name == 'negation':
            rels = r.get('relationships', [])
            neg_rels = [rel for rel in rels if isinstance(rel, dict) and
                       any(neg in rel.get('relation', '').lower()
                           for neg in ['not', 'refuse', 'reject', 'stopped', 'formerly'])]
            score += len(neg_rels) * 3

        # First person: check I->steven replacement
        if test_name == 'first_person':
            rels = r.get('relationships', [])
            steven_rels = [rel for rel in rels if isinstance(rel, dict) and
                          'steven' in rel.get('source', '').lower()]
            score += len(steven_rels) * 2

    # Relationship count bonus (capped)
    score += min(total_rels, 50) * 1.5
    # Fact count bonus (capped, penalize extremes)
    if 30 <= total_facts <= 60:
        score += 20
    elif total_facts > 100:
        score -= 20  # thinking model garbage

    # Speed bonus
    if total_time < 20:
        score += 15
    elif total_time < 30:
        score += 8
    elif total_time > 60:
        score -= 10

    return {
        'score': round(score, 1),
        'facts': total_facts,
        'rels': total_rels,
        'time': round(total_time, 1),
        'noise': f'{noise_correct}/{noise_total}',
    }


def main():
    print("=" * 80)
    print("CONFIG FINE-TUNING BENCHMARK")
    print("=" * 80)

    for model, configs in MODELS_TO_TUNE.items():
        # Check model exists
        try:
            r = requests.post(f'{OLLAMA_URL}/api/show', json={'model': model}, timeout=5)
            if r.status_code != 200:
                print(f"\nSKIPPED: {model} (not found)")
                continue
        except:
            print(f"\nSKIPPED: {model} (error)")
            continue

        # Unload other models
        for other_model in MODELS_TO_TUNE:
            if other_model != model:
                unload_model(other_model)

        print(f"\n{'=' * 80}")
        print(f"MODEL: {model}")
        print(f"{'=' * 80}")

        # Warm up once
        print("  Warming up...", end=' ', flush=True)
        warm_up(model)
        print("ready", flush=True)

        config_scores = []

        for cfg in configs:
            label = cfg['label']
            options = cfg['options']
            print(f"\n  --- {label}: {options} ---", flush=True)

            test_results = {}
            for test_name, test_data in TESTS.items():
                result = run_test(model, test_name, test_data, options)
                test_results[test_name] = result

                nf = len(result.get('facts', []))
                nr = len(result.get('relationships', []))
                t = result.get('fact_time', 0) + result.get('graph_time', 0) + result.get('noise_time', 0)
                status = ""
                if 'noise_kept' in result:
                    if not result['noise_kept']:
                        status = " DROPPED"
                    elif TESTS[test_name].get('expect_noise'):
                        status = " !!LEAK!!"
                    else:
                        status = " KEPT"

                # Multi-person detail
                mp_detail = ""
                if test_name == 'multi_person':
                    rels = result.get('relationships', [])
                    jake_devops = any(rel for rel in rels if isinstance(rel, dict) and
                                    'jake' in rel.get('source', '').lower() and
                                    'devops' in rel.get('target', '').lower())
                    mp_detail = f" jake->devops:{'YES' if jake_devops else 'NO'}"

                print(f"    [{test_name}] {nf}f/{nr}r ({t:.1f}s){status}{mp_detail}", flush=True)

            scores = score_result(test_results)
            config_scores.append((label, options, scores))
            print(f"    => SCORE: {scores['score']} | {scores['facts']}f/{scores['rels']}r | {scores['time']}s | noise:{scores['noise']}", flush=True)

        # Rank configs for this model
        config_scores.sort(key=lambda x: x[2]['score'], reverse=True)
        print(f"\n  RANKING for {model}:")
        for i, (label, options, scores) in enumerate(config_scores):
            marker = " <<<< BEST" if i == 0 else ""
            print(f"    {i+1}. {label}: score={scores['score']} | {scores['facts']}f/{scores['rels']}r | {scores['time']}s | noise:{scores['noise']}{marker}", flush=True)

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
