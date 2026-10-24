"""
Measure the supervisor's intent classification on eval/intent_queries.jsonl
(100 questions, 16-17 per intent, labeled by hand).

Needs an API key for the configured LLM_PROVIDER (see .env.example).

    python scripts/eval_intent.py
    python scripts/eval_intent.py --out eval/results/intent-gpt-4o.json

Prints accuracy, per-intent precision/recall, the confusion matrix and the
misclassified questions, and optionally writes them as JSON.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / "eval" / "intent_queries.jsonl"
INTENTS = ["SQL_QUERY", "VECTOR_QUERY", "GEO_QUERY", "ANALYTICS_QUERY", "COUNTERFACTUAL_QUERY", "HYBRID_QUERY"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(classify, rows: list[dict]) -> dict:
    """classify: question -> intent label."""
    preds = [classify(r["question"]) for r in rows]
    pairs = [(r["intent"], p) for r, p in zip(rows, preds)]
    confusion = Counter(pairs)
    per_intent = {}
    for label in INTENTS:
        tp = confusion[(label, label)]
        predicted = sum(n for (_, p), n in confusion.items() if p == label)
        actual = sum(n for (a, _), n in confusion.items() if a == label)
        per_intent[label] = {
            "precision": round(tp / predicted, 3) if predicted else None,
            "recall": round(tp / actual, 3) if actual else None,
            "support": actual,
        }
    return {
        "n": len(rows),
        "accuracy": round(sum(a == p for a, p in pairs) / len(rows), 4) if rows else None,
        "per_intent": per_intent,
        "confusion": {f"{a} -> {p}": n for (a, p), n in sorted(confusion.items())},
        "errors": [{"question": r["question"], "expected": r["intent"], "predicted": p}
                   for r, p in zip(rows, preds) if r["intent"] != p],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from langchain_core.messages import HumanMessage

    from medical_deserts.config import Config
    from medical_deserts.graph import SupervisorAgent

    supervisor = SupervisorAgent()
    result = evaluate(lambda q: supervisor({"messages": [HumanMessage(content=q)]})["intent"], load(args.set))
    result["model"] = f"{Config.LLM_PROVIDER}/{Config.LLM_MODEL}"

    print(f"\nModel: {result['model']}   n={result['n']}   accuracy={result['accuracy']:.1%}\n")
    for label, m in result["per_intent"].items():
        print(f"  {label:22s} precision={m['precision']}  recall={m['recall']}  n={m['support']}")
    print("\nMisclassified:")
    for e in result["errors"]:
        print(f"  [{e['expected']} -> {e['predicted']}] {e['question']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
