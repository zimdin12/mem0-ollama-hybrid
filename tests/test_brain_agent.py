"""
Phase 2 tests: Brain Agent Loop.

Tests the MemoryBrainAgent with real Ollama calls.
Requires: Ollama running with LLM_MODEL, test data in Qdrant.

Usage:
  # First, insert some test data
  curl -s -X POST http://localhost:8765/api/v1/memories/ \
    -H "Content-Type: application/json" \
    -d '{"text":"Steven uses an RTX 4090 with 24GB VRAM. He prefers dark mode. He builds games with Godot.","user_id":"brain_test_agent"}'

  # Then run tests
  docker cp tests/test_brain_agent.py openmemory-mcp:/usr/src/openmemory/tests/ && \
  MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/tests/test_brain_agent.py
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.brain.agent import MemoryBrainAgent, BrainResult

PASSED = 0
FAILED = 0
TEST_USER = "brain_test_agent"


def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS: {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        FAILED += 1


def run_tests():
    global PASSED, FAILED

    # Insert test data first
    print("\n=== Setup: Inserting test data ===")
    try:
        from app.brain.tools import vector_store, vector_search
        # Store test facts
        for fact in [
            "Brain test agent: Steven uses an RTX 4090 with 24GB VRAM",
            "Brain test agent: Steven prefers dark mode in all editors",
            "Brain test agent: Steven builds games with Godot Engine",
        ]:
            result = vector_store(fact, user_id=TEST_USER)
            print(f"  Stored: {result[:80]}")
        time.sleep(2)  # Wait for async graph extraction
    except Exception as e:
        print(f"  Setup warning: {e}")

    agent = MemoryBrainAgent()
    print(f"\nAgent config: model={agent.model}, url={agent.ollama_url}, max_steps={agent.max_steps}")

    print("\n=== Phase 2: Brain Agent Loop Tests ===\n")

    # --- Test 1: Read query ---
    print("Test group: Read queries")

    def test_ask_gpu():
        result = agent.run("What GPU does Steven use?", user_id=TEST_USER)
        assert result.success, f"Expected success, got error: {result.error}"
        assert result.answer, "Expected non-empty answer"
        assert len(result.tools_called) > 0, "Expected at least 1 tool call"
        assert "vector_search" in result.tools_called, f"Expected vector_search, got: {result.tools_called}"
        print(f"    Answer: {result.answer[:200]}")
        print(f"    Steps: {result.steps}, Tools: {result.tools_called}, Time: {result.elapsed_seconds:.1f}s")

    test("ask about GPU returns answer with vector_search", test_ask_gpu)

    # --- Test 2: Store operation ---
    print("\nTest group: Store operations")

    def test_store_memory():
        result = agent.run(
            "Remember that Steven recently started learning Zig programming language",
            user_id=TEST_USER,
        )
        assert result.success, f"Expected success, got error: {result.error}"
        assert result.answer, "Expected non-empty answer"
        assert "vector_store" in result.tools_called or "vector_search" in result.tools_called, \
            f"Expected vector_store or vector_search, got: {result.tools_called}"
        print(f"    Answer: {result.answer[:200]}")
        print(f"    Steps: {result.steps}, Tools: {result.tools_called}, Time: {result.elapsed_seconds:.1f}s")

    test("store operation stores or deduplicates", test_store_memory)

    # --- Test 3: Delete operation ---
    print("\nTest group: Delete operations")

    def test_delete():
        result = agent.run(
            "Delete all memories about Zig",
            user_id=TEST_USER,
        )
        # Should search, find, and delete
        assert result.answer, "Expected non-empty answer"
        print(f"    Answer: {result.answer[:200]}")
        print(f"    Steps: {result.steps}, Tools: {result.tools_called}, Time: {result.elapsed_seconds:.1f}s")

    test("delete operation searches then deletes", test_delete)

    # --- Test 4: Max steps respected ---
    print("\nTest group: Safety limits")

    def test_max_steps():
        limited_agent = MemoryBrainAgent()
        limited_agent.max_steps = 2
        result = limited_agent.run(
            "Do a very thorough analysis of everything about Steven including all relationships and connections",
            user_id=TEST_USER,
        )
        assert result.steps <= 2, f"Expected <= 2 steps, got {result.steps}"
        print(f"    Steps: {result.steps} (max_steps=2)")

    test("max_steps is respected", test_max_steps)

    # --- Test 5: No data found → early exit ---
    print("\nTest group: Early exit")

    def test_no_data_early_exit():
        result = agent.run(
            "What is the meaning of quantum chromodynamics in relation to Steven?",
            user_id=TEST_USER,
        )
        # Should exit reasonably quickly (not exhaust max_steps=12)
        assert result.steps <= 8, f"Expected <= 8 steps for near-empty search, got {result.steps}"
        print(f"    Answer: {result.answer[:200]}")
        print(f"    Steps: {result.steps} (early exit)")

    test("no data found exits early", test_no_data_early_exit)

    # --- Cleanup ---
    print("\n=== Cleanup ===")
    try:
        from app.brain.tools import vector_search, vector_delete
        hits = vector_search("brain test agent", limit=20, user_id=TEST_USER)
        if hits:
            ids = [h["id"] for h in hits]
            deleted = vector_delete(ids)
            print(f"  Cleaned up {deleted} test memories")
    except Exception as e:
        print(f"  Cleanup warning: {e}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'='*50}\n")
    return FAILED == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
