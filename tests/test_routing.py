import pytest
from langchain_core.messages import HumanMessage

from medical_deserts.agents.data_quality_router import DataQualityRouter
from medical_deserts.config import Config


@pytest.fixture
def router(monkeypatch, llm):
    monkeypatch.setattr(Config, "get_llm", classmethod(lambda cls: llm))
    return DataQualityRouter()


def plan(router, question):
    return router({"messages": [HumanMessage(content=question)]})["analytics_plan"]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Which facilities claim neurosurgery but lack an ICU?", ["SkillInfraAgent"]),
        ("How accessible is dialysis in rural Texas?", ["GeoAgent", "ReachabilityAgent"]),
        ("Are there systemic data quality patterns?", ["SkillInfraAgent", "ContradictionAgent"]),
        ("Where are the medical deserts?", ["GeoAgent", "ReachabilityAgent", "DesertTypologyAgent"]),
        ("Summarize the facilities", ["SkillInfraAgent"]),
    ],
)
def test_router_plans(router, question, expected):
    assert plan(router, question) == expected


def test_plan_has_no_duplicates(router):
    p = plan(router, "Which underserved areas lack equipment and have poor access and data quality patterns?")
    assert len(p) == len(set(p))


@pytest.mark.parametrize("intent, first_node", [
    ("SQL_QUERY", "DomainKnowledgeAgent"),
    ("HYBRID_QUERY", "DomainKnowledgeAgent"),
    ("GEO_QUERY", "GeoAgent"),
    ("ANALYTICS_QUERY", "DataQualityRouter"),
    ("COUNTERFACTUAL_QUERY", "CounterfactualEngine"),
])
def test_supervisor_intents_route_to_expected_node(graph_env, llm, intent, first_node):
    llm.intent = intent
    events = list(graph_env.get_graph().stream({"messages": [HumanMessage(content="q")], "analytics_executed": []},
                                                stream_mode="updates"))
    nodes = [next(iter(e)) for e in events]
    assert nodes[0] == "Supervisor"
    assert nodes[1] == first_node
    assert nodes[-1] == "ResponseAgent"


def test_unknown_intent_falls_back_to_sql(graph_env, llm):
    llm.intent = "SOMETHING_ELSE"
    events = list(graph_env.get_graph().stream({"messages": [HumanMessage(content="q")], "analytics_executed": []},
                                                stream_mode="updates"))
    assert [next(iter(e)) for e in events][1] == "DomainKnowledgeAgent"
