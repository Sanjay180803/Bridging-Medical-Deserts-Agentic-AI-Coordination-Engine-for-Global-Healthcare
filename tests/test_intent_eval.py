import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_intent", ROOT / "scripts" / "eval_intent.py")
eval_intent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_intent)


def test_intent_set_is_balanced_and_labeled():
    rows = eval_intent.load(eval_intent.DEFAULT_SET)
    assert len(rows) == 100
    counts = Counter(r["intent"] for r in rows)
    assert set(counts) == set(eval_intent.INTENTS)
    assert min(counts.values()) >= 16
    assert len({r["question"] for r in rows}) == 100


def test_evaluate_scores_a_classifier():
    rows = [{"question": "a", "intent": "SQL_QUERY"}, {"question": "b", "intent": "GEO_QUERY"},
            {"question": "c", "intent": "GEO_QUERY"}, {"question": "d", "intent": "HYBRID_QUERY"}]
    result = eval_intent.evaluate(lambda q: "GEO_QUERY" if q in "bc" else "SQL_QUERY", rows)
    assert result["accuracy"] == 0.75
    assert result["per_intent"]["GEO_QUERY"] == {"precision": 1.0, "recall": 1.0, "support": 2}
    assert result["per_intent"]["SQL_QUERY"]["precision"] == 0.5
    assert result["errors"] == [{"question": "d", "expected": "HYBRID_QUERY", "predicted": "SQL_QUERY"}]


def test_supervisor_prompt_is_what_gets_evaluated(graph_env, llm):
    from langchain_core.messages import HumanMessage
    from medical_deserts.graph import SupervisorAgent

    llm.intent = "GEO_QUERY"
    sup = SupervisorAgent()
    res = eval_intent.evaluate(lambda q: sup({"messages": [HumanMessage(content=q)]})["intent"],
                               [{"question": "Which hospitals are closest to Houston?", "intent": "GEO_QUERY"}])
    assert res["accuracy"] == 1.0
    assert "Which hospitals are closest to Houston?" in llm.prompts[-1]
