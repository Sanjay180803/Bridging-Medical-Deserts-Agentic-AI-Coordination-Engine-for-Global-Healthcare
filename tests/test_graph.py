import sqlite3

from langchain_core.messages import AIMessage, HumanMessage

from medical_deserts.state import latest_user_question


def test_latest_user_question_skips_ai_messages():
    state = {"messages": [HumanMessage(content="How many hospitals in CA?"), AIMessage(content="Query normalized.")]}
    assert latest_user_question(state) == "How many hospitals in CA?"


def test_sql_query_end_to_end(graph_env, llm):
    q = "How many hospitals are in California?"
    st = graph_env.run_query(q)

    assert st["intent"] == "SQL_QUERY"
    assert st["normalized_constraints"]["geography"]["states"] == ["CA"]
    assert st["sql_result"]["success"] is True
    assert int(st["sql_result"]["data"].iloc[0, 0]) == 15  # 15 CA hospitals in data/
    # Regression: the SQL and response prompts must see the user's question,
    # not the "Query normalized..." message DomainKnowledgeAgent appends.
    sql_prompt = next(p for p in llm.prompts if "SQL Agent" in p)
    assert f"ORIGINAL QUESTION: {q}" in sql_prompt
    assert st["final_response"].startswith(f"ANSWER TO: {q}")


def test_llm_generated_writes_cannot_modify_data(graph_env, llm, tmp_path):
    llm.sql = "DELETE FROM hospitals"
    st = graph_env.run_query("How many hospitals are in California?")
    assert st["sql_result"]["success"] is False
    with sqlite3.connect(tmp_path / "test.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM hospitals").fetchone()[0] == 150


def test_graph_is_built_once_per_process(graph_env):
    assert graph_env.get_graph() is graph_env.get_graph()


def test_desert_question_runs_full_analytics_pipeline(graph_env, llm):
    llm.intent = "ANALYTICS_QUERY"
    st = graph_env.run_query("Which regions are medical deserts with no coverage?")
    assert st["analytics_plan"] == ["GeoAgent", "ReachabilityAgent", "DesertTypologyAgent"]
    for agent in st["analytics_plan"]:
        assert agent in st["analytics_executed"], f"{agent} never ran"
    assert st["final_response"]
