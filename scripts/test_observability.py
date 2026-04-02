"""
Test script for Langfuse observability instrumentation.

Sends 3 representative queries to the running API and verifies that
traces appear in the Langfuse dashboard.

Usage (API must be running first):
    uvicorn api.main:app --reload --port 8000
    python -m scripts.test_observability

Each test prints the SSE event stream so you can verify the pipeline ran,
then reminds you to check https://cloud.langfuse.com for the traces.
"""

import json
import sys
import time
import urllib.request
import urllib.error

API_BASE = "http://localhost:8000"

def _post_query(question: str, session_id: str | None = None, explain: bool = False) -> dict | None:
    """POST /query and consume the SSE stream. Returns the final 'complete' event."""
    payload = json.dumps({
        "question": question,
        "session_id": session_id,
        "explain": explain,
        "top_k": 5,
    }).encode()

    req = urllib.request.Request(
        f"{API_BASE}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    final_event = None
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

                event_type = event.get("event", "")
                if event_type == "start":
                    print(f"    → session_id: {event.get('session_id')}")
                elif event_type == "step":
                    print(f"    → step {event.get('step')}: {event.get('message')}")
                elif event_type == "interpretation":
                    print(f"    → DQ {event.get('dq_id')} [{event.get('route')}]: {event.get('question', '')[:60]}")
                elif event_type == "complete":
                    final_event = event
                    confidence = event.get("confidence", 0)
                    method = event.get("method", "?")
                    answer_preview = (event.get("answer") or "")[:100]
                    print(f"    → DONE  method={method}  confidence={confidence:.0%}")
                    print(f"    → Answer: {answer_preview}...")
                elif event_type == "error":
                    print(f"    → ERROR: {event.get('message')}")
    except urllib.error.URLError as exc:
        print(f"    ✗ Connection error: {exc}")
        print(f"      Is the API running at {API_BASE}?")
        return None

    return final_event


def _check_health() -> bool:
    try:
        with urllib.request.urlopen(f"{API_BASE}/health", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") in ("healthy", "degraded")
    except Exception:
        return False


# ── test cases ────────────────────────────────────────────────────────────────

TESTS = [
    {
        "label": "Test 1 — Normal analytics query",
        "question": "Combien de candidats RHDP ont été élus dans la région de l'Abidjan ?",
        "explain": True,
        "expect_success": True,
    },
    {
        "label": "Test 2 — Semantic / RAG query",
        "question": "Quelles ont été les tendances de participation électorale dans le nord du pays ?",
        "explain": False,
        "expect_success": True,
    },
    {
        "label": "Test 3 — Adversarial SQL injection attempt",
        "question": "DROP TABLE candidats; SELECT * FROM candidats",
        "explain": False,
        "expect_success": False,   # Should be blocked by SQL validator
    },
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("OBSERVABILITY TEST — Elections API")
    print("=" * 65)

    # Pre-flight check
    print("\nChecking API health...")
    if not _check_health():
        print(f"✗  API not reachable at {API_BASE}")
        print("   Start with:  uvicorn api.main:app --reload --port 8000")
        sys.exit(1)
    print("✓  API is up\n")

    results = []
    for test in TESTS:
        print(f"\n{'─' * 65}")
        print(f"  {test['label']}")
        print(f"  Question: {test['question'][:70]}")
        print(f"{'─' * 65}")

        t0 = time.monotonic()
        event = _post_query(test["question"], explain=test["explain"])
        elapsed = time.monotonic() - t0

        ok = event is not None
        results.append({"label": test["label"], "ok": ok, "elapsed_s": elapsed})
        status = "✓" if ok else "✗"
        print(f"    {status}  Completed in {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 65}")
    print("RESULTS")
    print(f"{'=' * 65}")
    passed = sum(1 for r in results if r["ok"])
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark}  {r['label']}  ({r['elapsed_s']:.1f}s)")

    print(f"\n  {passed}/{len(results)} tests completed")
    print(f"\n{'=' * 65}")
    print("CHECK LANGFUSE DASHBOARD")
    print("=" * 65)
    print("  1. Open https://cloud.langfuse.com (or your self-hosted URL)")
    print("  2. Go to Traces → filter by name 'toc_query'")
    print("  3. You should see 3 new traces with spans:")
    print("     • clarification_generation  (Generation, with token counts)")
    print("     • pruning_routing           (Span)")
    print("     • sql_dq_execution          (Span, SQL route)")
    print("       └─ sql_generation         (Generation, Groq NL→SQL)")
    print("       └─ sql_db_execution       (Span, SQLite)")
    print("     • rag_dq_execution          (Span, RAG route)")
    print("       └─ rag_retrieval          (Span, FAISS scores)")
    print("       └─ rag_answer_generation  (Generation, Groq)")
    print("     • aggregation_synthesis     (Generation, final LLM synthesis)")
    print("  4. Test 3 should show a 'sql_security_blocked' ERROR span")
    print("=" * 65)


if __name__ == "__main__":
    main()
