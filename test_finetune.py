#!/usr/bin/env python3
"""
Focused finetune tests — targets specific v2 weaknesses.
Tests: step efficiency, update intent, delete batching, search quality.
"""

import json
import sys
import time
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API = "http://localhost:8765"
USER = "finetune_test"

def brain(request, label=""):
    start = time.time()
    r = requests.post(f"{API}/api/v1/brain",
        json={"request": request, "user_id": USER}, timeout=300)
    elapsed = time.time() - start
    data = r.json()
    steps = data.get("steps", "?")
    tools = data.get("tools_called", [])
    answer = data.get("answer", "")[:200]
    success = data.get("success", False)

    print(f"\n  [{elapsed:.1f}s] {label}")
    print(f"    Steps: {steps}, Tools: {tools}")
    print(f"    Success: {success}")
    print(f"    Answer: {answer}")

    return data, elapsed

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run():
    print(f"API: {API}, User: {USER}")

    # Setup: create user and store some facts
    header("SETUP: Store baseline facts")

    brain("Remember: Steven uses an RTX 4090 GPU with 24GB VRAM", "Store GPU fact")
    brain("Remember: Steven prefers dark mode in all applications", "Store dark mode fact")
    brain("Remember: Steven produces synthwave music using Ableton Live", "Store music fact")
    brain("Remember: Steven has a cat named Pixel", "Store cat fact")
    brain("Remember: Echoes of the Fallen uses 32x32x32 chunks", "Store chunk fact")

    time.sleep(3)  # let graph extract

    # ================================================================
    # TEST 1: Simple search step efficiency
    # ================================================================
    header("TEST 1: Search step efficiency (target: 2 steps)")

    data, elapsed = brain("What GPU does Steven use?", "Simple search")
    steps = data.get("steps", 99)
    tools = data.get("tools_called", [])

    if steps <= 2:
        print("    PASS: 2 steps or less")
    elif steps <= 3:
        print(f"    OK: {steps} steps (acceptable)")
    else:
        print(f"    FAIL: {steps} steps (too many)")

    # Check if it unnecessarily queries graph
    graph_calls = [t for t in tools if t == "graph_query"]
    if len(graph_calls) == 0:
        print("    PASS: No unnecessary graph_query")
    else:
        print(f"    WARN: {len(graph_calls)} graph_query calls (possibly unnecessary)")

    # ================================================================
    # TEST 2: Store step efficiency
    # ================================================================
    header("TEST 2: Store step efficiency (target: 2-3 steps)")

    data, elapsed = brain("Remember: Steven recently started learning Zig programming", "Store new fact")
    steps = data.get("steps", 99)

    if steps <= 3:
        print(f"    PASS: {steps} steps")
    else:
        print(f"    FAIL: {steps} steps (target is 2-3)")

    # ================================================================
    # TEST 3: UPDATE intent — must delete old + store new
    # ================================================================
    header("TEST 3: Update intent (must delete old, store new)")

    data, elapsed = brain("Steven upgraded his GPU from RTX 4090 to RTX 5090 with 32GB VRAM", "Update GPU")
    steps = data.get("steps", 99)
    tools = data.get("tools_called", [])

    has_search = "vector_search" in tools
    has_delete = "vector_delete" in tools
    has_store = "vector_store" in tools

    print(f"    Search: {'YES' if has_search else 'NO'}")
    print(f"    Delete: {'YES' if has_delete else 'NO'}")
    print(f"    Store:  {'YES' if has_store else 'NO'}")

    if has_search and has_delete and has_store:
        print("    PASS: Full update cycle (search + delete + store)")
    elif has_search and has_store and not has_delete:
        print("    FAIL: Stored additive — did NOT delete old fact")
    else:
        print(f"    WARN: Unexpected tool pattern: {tools}")

    # Verify: search should return 5090, not 4090
    data2, _ = brain("What GPU does Steven use?", "Verify update")
    answer = data2.get("answer", "").lower()
    if "5090" in answer and "4090" not in answer:
        print("    PASS: Answer says 5090, no 4090")
    elif "5090" in answer and "4090" in answer:
        print("    WARN: Both 4090 and 5090 in answer (old not fully deleted)")
    elif "4090" in answer:
        print("    FAIL: Still says 4090 — update didn't work")
    else:
        print(f"    UNCLEAR: {answer[:100]}")

    # ================================================================
    # TEST 4: Batch delete — single vector_delete call
    # ================================================================
    header("TEST 4: Delete efficiency (target: single delete call)")

    # Store 2 extra music facts first
    brain("Remember: Steven uses a Focusrite Scarlett 2i2 audio interface", "Store audio fact 1")
    brain("Remember: Steven's music leans toward dark ambient genres", "Store audio fact 2")
    time.sleep(2)

    data, elapsed = brain("Delete all memories about music and audio", "Batch delete")
    steps = data.get("steps", 99)
    tools = data.get("tools_called", [])

    delete_calls = tools.count("vector_delete")
    print(f"    Delete calls: {delete_calls}")

    if delete_calls == 1:
        print("    PASS: Single batch delete call")
    elif delete_calls <= 2:
        print(f"    OK: {delete_calls} delete calls (acceptable)")
    else:
        print(f"    FAIL: {delete_calls} delete calls (should batch them)")

    if steps <= 4:
        print(f"    PASS: {steps} steps (efficient)")
    else:
        print(f"    WARN: {steps} steps (could be faster)")

    # ================================================================
    # TEST 5: Early exit on no data
    # ================================================================
    header("TEST 5: Early exit on empty results (target: 2-3 steps)")

    data, elapsed = brain("What does Steven think about quantum computing?", "No-data query")
    steps = data.get("steps", 99)
    answer = data.get("answer", "").lower()

    if steps <= 3:
        print(f"    PASS: {steps} steps (fast exit)")
    elif steps <= 5:
        print(f"    OK: {steps} steps")
    else:
        print(f"    FAIL: {steps} steps (should exit early)")

    if "no memor" in answer or "not found" in answer or "no information" in answer or "no record" in answer:
        print("    PASS: Correctly reports no data found")
    else:
        print(f"    WARN: Answer doesn't clearly state no data: {answer[:100]}")

    # ================================================================
    # TEST 6: Multi-fact store
    # ================================================================
    header("TEST 6: Multi-fact store (2 facts in one request)")

    data, elapsed = brain(
        "Remember these two facts: Steven adopted a second cat named Luna. Steven started learning Japanese.",
        "Multi-fact store"
    )
    steps = data.get("steps", 99)
    tools = data.get("tools_called", [])
    store_calls = tools.count("vector_store")

    print(f"    Store calls: {store_calls}")
    if store_calls >= 2:
        print("    PASS: Stored both facts")
    else:
        print(f"    WARN: Only {store_calls} store calls (expected 2)")

    if steps <= 6:
        print(f"    PASS: {steps} steps")
    else:
        print(f"    WARN: {steps} steps")

    # ================================================================
    # SUMMARY
    # ================================================================
    header("SUMMARY")
    print("  Check results above for PASS/FAIL/WARN on each test.")
    print("  Key metrics to watch:")
    print("    - Simple search: <= 2 steps, no unnecessary graph_query")
    print("    - Simple store: <= 3 steps")
    print("    - Update: search + delete + store (3 tools)")
    print("    - Batch delete: single vector_delete call")
    print("    - Early exit: <= 3 steps")


if __name__ == "__main__":
    run()
