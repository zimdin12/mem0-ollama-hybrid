"""
Phase 1 tests: Brain Tools — low-level DB access layer.

Usage:
  docker cp tests/test_brain_tools.py openmemory-mcp:/usr/src/openmemory/tests/ && \
  MSYS_NO_PATHCONV=1 docker exec openmemory-mcp python3 /usr/src/openmemory/tests/test_brain_tools.py
"""

import json
import sys
import os

# Ensure app modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.brain.tools import (
    vector_search, vector_store, vector_delete,
    graph_query, graph_mutate,
    sql_query, sql_mutate,
    embed,
)

PASSED = 0
FAILED = 0
TEST_USER = "brain_test_user"


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
    print("\n=== Phase 1: Brain Tools Tests ===\n")

    # --- embed ---
    print("Tool: embed")

    def test_embed_returns_1024():
        result = embed("Steven uses an RTX 4090")
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 1024, f"Expected 1024 dims, got {len(result)}"
        assert all(isinstance(x, float) for x in result[:10]), "Expected floats"

    test("embed returns 1024 floats", test_embed_returns_1024)

    # --- vector_search ---
    print("\nTool: vector_search")

    def test_vector_search_returns_list():
        results = vector_search("test query", limit=5, user_id=TEST_USER)
        assert isinstance(results, list), f"Expected list, got {type(results)}"

    test("vector_search returns list", test_vector_search_returns_list)

    # --- vector_store + search + delete ---
    print("\nTool: vector_store → search → delete")

    stored_id = None

    def test_vector_store():
        nonlocal stored_id
        result = vector_store("Brain test: Steven's favorite color is blue", user_id=TEST_USER)
        assert result and not result.startswith("Error"), f"Store failed: {result}"
        if not result.startswith("DUPLICATE"):
            stored_id = result
        else:
            # Already exists from previous run — find it
            hits = vector_search("Steven's favorite color is blue", limit=1, user_id=TEST_USER)
            if hits:
                stored_id = hits[0]["id"]

    test("vector_store stores a fact", test_vector_store)

    def test_vector_search_finds_stored():
        results = vector_search("favorite color", limit=5, user_id=TEST_USER)
        assert len(results) > 0, "Expected at least 1 result"
        found = any("blue" in r.get("text", "").lower() or "color" in r.get("text", "").lower() for r in results)
        assert found, f"Expected to find 'blue'/'color' in results: {results}"

    test("vector_search finds stored fact", test_vector_search_finds_stored)

    def test_vector_store_dedup():
        result = vector_store("Brain test: Steven's favorite color is blue", user_id=TEST_USER)
        assert "DUPLICATE" in result, f"Expected DUPLICATE, got: {result}"

    test("vector_store detects duplicate", test_vector_store_dedup)

    def test_vector_delete():
        if not stored_id:
            raise Exception("No stored_id from previous test")
        count = vector_delete([stored_id])
        assert count >= 1, f"Expected >= 1 deleted, got {count}"

    test("vector_delete removes stored fact", test_vector_delete)

    # --- graph_query ---
    print("\nTool: graph_query")

    def test_graph_query_read_only():
        results = graph_query(
            "MATCH (n) RETURN n.name LIMIT 3",
            user_id=TEST_USER,
        )
        assert isinstance(results, list), f"Expected list, got {type(results)}"

    test("graph_query returns list", test_graph_query_read_only)

    def test_graph_query_rejects_mutating():
        try:
            graph_query("CREATE (n:Test {name: 'bad'})")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "read-only" in str(e).lower() or "mutating" in str(e).lower()

    test("graph_query rejects mutating Cypher", test_graph_query_rejects_mutating)

    # --- graph_mutate ---
    print("\nTool: graph_mutate")

    def test_graph_mutate_creates_and_deletes():
        # Create test node
        created = graph_mutate(
            "CREATE (n:BrainTest {name: 'brain_test_node', user_id: 'brain_test'}) RETURN n",
            user_id=TEST_USER,
        )
        assert created >= 1, f"Expected >= 1 created, got {created}"

        # Clean up
        deleted = graph_mutate(
            "MATCH (n:BrainTest {name: 'brain_test_node'}) DETACH DELETE n",
            user_id=TEST_USER,
        )
        assert deleted >= 1, f"Expected >= 1 deleted, got {deleted}"

    test("graph_mutate creates and deletes node", test_graph_mutate_creates_and_deletes)

    # --- sql_query ---
    print("\nTool: sql_query")

    def test_sql_query_reads():
        results = sql_query("SELECT COUNT(*) as cnt FROM brain_audit")
        assert isinstance(results, list), f"Expected list, got {type(results)}"
        assert len(results) == 1, f"Expected 1 row, got {len(results)}"
        assert "cnt" in results[0], f"Expected 'cnt' column, got {results[0].keys()}"

    test("sql_query reads brain_audit", test_sql_query_reads)

    def test_sql_query_rejects_mutating():
        try:
            sql_query("DELETE FROM brain_audit WHERE 1=0")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "read-only" in str(e).lower() or "mutating" in str(e).lower()

    test("sql_query rejects mutating SQL", test_sql_query_rejects_mutating)

    # --- sql_mutate ---
    print("\nTool: sql_mutate")

    def test_sql_mutate():
        # Insert test row
        affected = sql_mutate(
            "INSERT INTO brain_audit (tool, user_id, input) VALUES (:tool, :uid, :input)",
            {"tool": "test", "uid": TEST_USER, "input": "test data"},
        )
        assert affected == 1, f"Expected 1 affected, got {affected}"

        # Clean up
        sql_mutate(
            "DELETE FROM brain_audit WHERE tool = 'test' AND user_id = :uid",
            {"uid": TEST_USER},
        )

    test("sql_mutate inserts and deletes", test_sql_mutate)

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'='*50}\n")
    return FAILED == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
