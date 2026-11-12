"""
Small evaluation script for the RAG assistant.

Run it from the project folder (GEMINI_API_KEY must be set, PDFs in ./pdfs):
    python evaluate.py

It reads eval_questions.json, asks every question, and reports:
  - answer accuracy on answerable questions (expected keywords appear in the answer)
  - refusal accuracy on unanswerable questions (the assistant says it doesn't know)
  - retrieval hits (the expected page was among the sources), when a page is given
Results are saved to eval_results.json and eval_report.md (paste it into your README).
"""
import json
import time
from pathlib import Path

import rag_settings as config

config.setup_logging()  # must run before rag_engine is imported

from rag_engine import ask, index_documents  # noqa: E402

HERE = Path(__file__).parent
QUESTIONS_FILE = HERE / "eval_questions.json"
RESULTS_FILE = HERE / "eval_results.json"
REPORT_FILE = HERE / "eval_report.md"

PAUSE_SECONDS = 2  # pause between questions to stay within API rate limits
REFUSAL_MARKER = "enough information"  # the assistant's "I don't know" wording


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in answer.lower()


def is_template(item: dict) -> bool:
    """Template rows still contain the word REPLACE and are skipped."""
    return "REPLACE" in json.dumps(item)


def evaluate_item(item: dict) -> dict:
    result = ask(item["question"])
    answer, sources = result["answer"], result["sources"]
    notes = []

    if item["type"] == "answerable":
        refused = is_refusal(answer)
        missing = [
            k for k in item.get("expected_keywords", []) if k.lower() not in answer.lower()
        ]
        passed = not refused and not missing
        if refused:
            notes.append("refused to answer")
        if missing:
            notes.append(f"missing keywords: {missing}")

        page_hit = None
        page = item.get("expected_page")
        if page is not None:
            page_hit = any(f"(page {page})" in s for s in sources)
            if not page_hit:
                notes.append(f"page {page} not among retrieved sources")
    else:  # unanswerable: the assistant should say it doesn't know
        passed = is_refusal(answer)
        page_hit = None
        if not passed:
            notes.append("answered a question the documents don't cover")

    return {
        "question": item["question"],
        "type": item["type"],
        "passed": passed,
        "page_hit": page_hit,
        "notes": notes,
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
    }


def percent(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def main():
    items = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    real_items = [it for it in items if not is_template(it)]
    skipped = len(items) - len(real_items)

    if skipped:
        print(f"Skipping {skipped} template row(s) that still contain 'REPLACE'.")
    if not real_items:
        print("No real questions found. Edit eval_questions.json first.")
        return

    index_documents()

    results = []
    for i, item in enumerate(real_items, start=1):
        res = evaluate_item(item)
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        print(f"[{i}/{len(real_items)}] {mark} ({res['type']}): {res['question']}")
        for note in res["notes"]:
            print(f"      - {note}")
        time.sleep(PAUSE_SECONDS)

    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    ans_ok = sum(r["passed"] for r in answerable)
    ref_ok = sum(r["passed"] for r in unanswerable)
    with_page = [r for r in answerable if r["page_hit"] is not None]
    page_ok = sum(r["page_hit"] for r in with_page)
    total_ok = ans_ok + ref_ok

    summary = [
        f"Overall: {percent(total_ok, len(results))}",
        f"Answerable questions answered correctly: {percent(ans_ok, len(answerable))}",
        f"Out-of-scope questions correctly refused: {percent(ref_ok, len(unanswerable))}",
        f"Retrieval hit (expected page in sources): {percent(page_ok, len(with_page))}",
    ]
    print("\n" + "\n".join(summary))

    RESULTS_FILE.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    failures = [r for r in results if not r["passed"]]
    lines = ["## Evaluation", "", f"Tested on {len(results)} hand-written questions.", ""]
    lines += [f"- {s}" for s in summary]
    if failures:
        lines += ["", "### Failures", ""]
        lines += [f"- {r['question']} ({'; '.join(r['notes'])})" for r in failures]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved {RESULTS_FILE.name} and {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
