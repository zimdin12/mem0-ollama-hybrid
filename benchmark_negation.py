#!/usr/bin/env python3
"""Deep investigation: qwen3.5 negation handling, thinking mode, temperature sweep."""
import requests, json, time

OLLAMA = 'http://ollama:11434'

NEGATION_TEXT = "Steven does NOT use Visual Studio Code anymore. He switched from VS Code to Neovim last month. He tried Emacs but didn't like it. He has never used IntelliJ."

GRAPH_SYS = """Extract entities and relationships from text. Return ONLY a JSON object.
Format: {"entities": [{"name": "...", "type": "..."}, ...], "relationships": [{"source": "...", "relation": "...", "target": "..."}, ...]}
Entity types: person, project, technology, tool, framework, language, hardware, organization, place, game_element, algorithm, database, service
Only person, project, and game_element entities should be relationship sources.

IMPORTANT: You MUST generate relationships for every entity. Every entity should connect to at least one person or project via a relationship. If an entity has no relationship, remove it from entities too.
IMPORTANT: Preserve NEGATIONS in relationship names. If someone does NOT use something, the relation should be "does_not_use" or "stopped_using", NOT "uses"."""

EXTENDED_TESTS = {
    "negation": NEGATION_TEXT,
    "preferences": "Steven prefers Arch Linux over Ubuntu. He dislikes Windows but uses it for gaming. He refuses to use macOS.",
    "temporal": "Steven used to work at Google but left in 2024. He now works at a startup called NordicAI. Before Google he was at Microsoft for 2 years.",
    "first_person": "I built a memory system using Qdrant and Neo4j. I replaced tool calling with JSON mode because it was unreliable. My favorite editor is Neovim.",
    "complex_neg": "The project does NOT support multiplayer yet. Steven rejected Unity because of poor Linux support. They considered using Godot but chose Unreal instead. The team has NOT hired a sound designer.",
}


def strip_fences(text):
    text = text.strip()
    if text.startswith('```'):
        nl = text.find('\n')
        if nl != -1: text = text[nl + 1:]
        if text.rstrip().endswith('```'): text = text.rstrip()[:-3].rstrip()
    return text


def test_negation(model, temp, think_mode, text=NEGATION_TEXT, sys_prompt=GRAPH_SYS):
    """Test graph extraction with specific params."""
    opts = {'temperature': temp, 'top_p': 0.8, 'top_k': 20}
    if 'qwen3.5' in model.lower() or 'qwen35' in model.lower():
        opts['presence_penalty'] = 1.5

    start = time.time()
    r = requests.post(f'{OLLAMA}/api/chat', json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f'Extract from:\n{text}'},
        ],
        'stream': False, 'format': 'json', 'think': think_mode,
        'options': opts,
    }, timeout=120)
    elapsed = time.time() - start

    resp = r.json()
    msg = resp.get('message', {})
    content = msg.get('content', '') if isinstance(msg, dict) else str(msg)

    # Check for thinking tokens in content
    has_thinking = '<think>' in content or '</think>' in content
    thinking_text = ''
    if has_thinking:
        import re
        m = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if m:
            thinking_text = m.group(1)[:200]
            # Strip thinking from content for JSON parsing
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    try:
        g = json.loads(strip_fences(content))
        rels = g.get('relationships', [])
    except:
        rels = []

    return {
        'elapsed': elapsed,
        'rels': rels,
        'has_thinking': has_thinking,
        'thinking_text': thinking_text,
        'raw_len': len(content),
    }


def unload_all():
    ps = requests.get(f'{OLLAMA}/api/ps').json()
    for m in ps.get('models', []):
        requests.post(f'{OLLAMA}/api/generate', json={'model': m['name'], 'keep_alive': 0})
    time.sleep(2)


def warmup(model):
    requests.post(f'{OLLAMA}/api/chat', json={
        'model': model, 'messages': [{'role': 'user', 'content': 'Ready'}],
        'stream': False, 'think': False,
    }, timeout=120)


if __name__ == '__main__':
    model = 'qwen3.5:4b'

    # ============================================================
    # Test 1: Temperature sweep on negation
    # ============================================================
    print('=' * 70)
    print('TEST 1: Temperature sweep on negation (qwen3.5:4b, think=False)')
    print('=' * 70)

    unload_all()
    warmup(model)

    for temp in [0.1, 0.3, 0.5, 0.7, 1.0]:
        result = test_negation(model, temp, False)
        rels_summary = []
        for rel in result['rels']:
            s = rel.get('source', '?')
            r = rel.get('relation', '?')
            t = rel.get('target', '?')
            rels_summary.append(f'{s}--{r}-->{t}')
        print(f'\n  temp={temp}: {len(result["rels"])}r ({result["elapsed"]:.1f}s) thinking={result["has_thinking"]}')
        for rs in rels_summary:
            neg_marker = ' <<<WRONG' if ('visual_studio' in rs and ('uses-->' in rs or 'use-->' in rs) and 'not' not in rs and 'stop' not in rs) else ''
            print(f'    {rs}{neg_marker}')

    # ============================================================
    # Test 2: Think mode variations
    # ============================================================
    print('\n' + '=' * 70)
    print('TEST 2: Think mode variations (qwen3.5:4b, temp=0.7)')
    print('=' * 70)

    for think in [False, True]:
        label = 'think=True' if think else 'think=False'
        result = test_negation(model, 0.7, think)
        print(f'\n  {label}: {len(result["rels"])}r ({result["elapsed"]:.1f}s)')
        if result['has_thinking']:
            print(f'    THINKING: {result["thinking_text"][:150]}...')
        for rel in result['rels']:
            s, r, t = rel.get('source','?'), rel.get('relation','?'), rel.get('target','?')
            neg_marker = ' <<<WRONG' if ('visual_studio' in t and ('uses' == r or 'use' == r) and 'not' not in r and 'stop' not in r) else ''
            print(f'    {s} --{r}--> {t}{neg_marker}')

    # ============================================================
    # Test 3: qwen3 vs qwen3.5 on all extended tests (temp=0.7)
    # ============================================================
    print('\n' + '=' * 70)
    print('TEST 3: Extended tests — qwen3 vs qwen3.5 (temp=0.7, think=False)')
    print('=' * 70)

    for m in ['qwen3:4b-instruct-2507-q4_K_M', 'qwen3.5:4b']:
        short = m.split(':')[0]
        unload_all()
        warmup(m)

        print(f'\n  --- {short} ---')
        for test_name, text in EXTENDED_TESTS.items():
            result = test_negation(m, 0.7, False, text)
            print(f'\n  [{test_name}] {len(result["rels"])}r ({result["elapsed"]:.1f}s)')
            for rel in result['rels']:
                s, r, t = rel.get('source','?'), rel.get('relation','?'), rel.get('target','?')
                # Flag wrong negation
                flag = ''
                if 'not' in text.lower() or 'reject' in text.lower() or 'refuse' in text.lower():
                    if r in ('uses', 'use', 'works_at', 'supports') and any(neg in text.lower() for neg in ['not use', 'not support', 'reject', 'refuse', 'left', 'dislike']):
                        if t in text.lower().split('not')[0] if 'not' in text.lower() else False:
                            flag = ' ?'
                print(f'    {s} --{r}--> {t}{flag}')

    # ============================================================
    # Test 4: Context length check
    # ============================================================
    print('\n' + '=' * 70)
    print('TEST 4: Actual context used by Ollama')
    print('=' * 70)

    for m in ['qwen3:4b-instruct-2507-q4_K_M', 'qwen3.5:4b']:
        r = requests.post(f'{OLLAMA}/api/show', json={'model': m})
        d = r.json()
        params = d.get('parameters', '')
        info = d.get('model_info', {})
        ctx_keys = [k for k in info if 'context' in k.lower()]
        print(f'\n  {m}:')
        for k in ctx_keys:
            print(f'    {k}: {info[k]}')
        # Check if num_ctx is set in params
        if 'num_ctx' in params:
            print(f'    num_ctx from params: found')
        else:
            print(f'    num_ctx: NOT SET (uses model default)')
        print(f'    Parameters: {params[:200]}')
