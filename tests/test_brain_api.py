"""
Phase 3 tests: Brain REST API endpoints.

Tests the /api/v1/brain/* endpoints via HTTP.
Run from the host machine (not inside the container).

Usage:
  python tests/test_brain_api.py [--api-url http://localhost:8765]
"""

import json
import sys
import time
import requests

# Fix Unicode output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API_URL = "http://localhost:8765"
TEST_USER = "brain_api_test"

# Parse --api-url argument
for i, arg in enumerate(sys.argv):
    if arg == "--api-url" and i + 1 < len(sys.argv):
        API_URL = sys.argv[i + 1]

PASSED = 0
FAILED = 0


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

    # Check API is up
    print(f"\nAPI URL: {API_URL}")
    try:
        resp = requests.get(f"{API_URL}/docs", timeout=5)
        print(f"API status: {resp.status_code}\n")
    except Exception as e:
        print(f"API not reachable: {e}")
        print("Start the container first: docker compose up -d openmemory-mcp")
        sys.exit(1)

    # Insert test data — use the brain's own store endpoint so user gets auto-created
    print("=== Setup: Inserting test data ===")
    try:
        resp = requests.post(f"{API_URL}/api/v1/brain/do", json={
            "request": "Remember these facts: Steven's favorite IDE is VSCode. He uses it for TypeScript and Python.",
            "user_id": TEST_USER,
            "confirmed": True,
        }, timeout=120)
        print(f"  Insert via brain/do: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Brain stored: {data.get('answer', '')[:200]}")
        time.sleep(3)  # Wait for graph extraction
    except Exception as e:
        print(f"  Setup warning: {e}")

    print("\n=== Phase 3: Brain REST API Tests ===\n")

    # --- /brain/status ---
    print("Endpoint: GET /brain/status")

    def test_status():
        resp = requests.get(f"{API_URL}/api/v1/brain/status")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "model" in data, f"Missing 'model' in response: {data}"
        assert "tools_available" in data, f"Missing 'tools_available' in response: {data}"
        print(f"    Model: {data['model']}, Tools: {len(data['tools_available'])}")

    test("status returns model info", test_status)

    # --- /brain/ask ---
    print("\nEndpoint: POST /brain/ask")

    def test_ask_basic():
        resp = requests.post(f"{API_URL}/api/v1/brain/ask", json={
            "request": "What IDE does Steven use?",
            "user_id": TEST_USER,
        }, timeout=120)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "answer" in data, f"Missing 'answer' in response"
        assert data["success"], f"Expected success=true, got: {data}"
        assert data["steps"] > 0, f"Expected steps > 0"
        print(f"    Answer: {data['answer'][:200]}")
        print(f"    Steps: {data['steps']}, Tools: {data['tools_called']}, Time: {data.get('elapsed_seconds', '?')}s")

    test("ask returns answer with 200", test_ask_basic)

    # --- /brain/do (non-destructive) ---
    print("\nEndpoint: POST /brain/do (store)")

    def test_do_store():
        resp = requests.post(f"{API_URL}/api/v1/brain/do", json={
            "request": "Remember that Steven recently upgraded to 96GB of RAM",
            "user_id": TEST_USER,
            "confirmed": True,
        }, timeout=120)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "answer" in data, f"Missing 'answer' in response"
        print(f"    Answer: {data['answer'][:200]}")
        print(f"    Steps: {data['steps']}, Tools: {data['tools_called']}")

    test("do (store) executes and returns answer", test_do_store)

    # --- /brain/do (destructive, unconfirmed) ---
    print("\nEndpoint: POST /brain/do (delete, unconfirmed)")

    def test_do_delete_unconfirmed():
        resp = requests.post(f"{API_URL}/api/v1/brain/do", json={
            "request": "Delete all memories about RAM",
            "user_id": TEST_USER,
            "confirmed": False,
        }, timeout=120)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        # May or may not require confirmation depending on whether brain used destructive tools
        if data.get("requires_confirmation"):
            assert data.get("planned_actions"), "Expected planned_actions when requires_confirmation"
            print(f"    Requires confirmation: {len(data['planned_actions'])} planned actions")
        else:
            print(f"    No destructive actions (brain may have found nothing to delete)")
        print(f"    Steps: {data['steps']}, Tools: {data['tools_called']}")

    test("do (delete, unconfirmed) returns plan or executes read-only part", test_do_delete_unconfirmed)

    # --- /brain/audit ---
    print("\nEndpoint: GET /brain/audit")

    def test_audit():
        resp = requests.get(f"{API_URL}/api/v1/brain/audit", params={
            "user_id": TEST_USER,
            "limit": 10,
        })
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert "rows" in data, f"Missing 'rows' in response"
        assert data["count"] > 0, f"Expected audit rows > 0, got {data['count']}"
        print(f"    Audit rows: {data['count']}")

    test("audit returns rows", test_audit)

    # --- Cleanup ---
    print("\n=== Cleanup ===")
    try:
        resp = requests.post(f"{API_URL}/api/v1/memories/delete_all", json={"user_id": TEST_USER})
        print(f"  Cleanup: {resp.status_code} — {resp.json().get('message', '')}")
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
