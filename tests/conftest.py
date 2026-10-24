"""
Test fixtures. No test calls a real LLM or embedding API: a scripted chat
model answers each agent's prompt deterministically, and the vector store
uses a local hashing embedding function.
"""
import json
import re
from typing import Any, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedLLM(BaseChatModel):
    """Answers by recognizing which agent's prompt it received.

    intent: what the supervisor returns.
    sql:    what the SQL agent's generation step returns.
    """

    intent: str = "SQL_QUERY"
    sql: str = "SELECT COUNT(DISTINCT pk_unique_id) AS n FROM hospitals WHERE address_stateOrRegion = 'CA'"
    prompts: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kw: Any) -> ChatResult:
        prompt = messages[-1].content if messages else ""
        self.prompts.append(prompt)
        if "Supervisor Agent" in prompt:
            out = self.intent
        elif "Domain Knowledge Agent" in prompt:
            out = json.dumps({"entity": "hospital", "geography": {"states": ["CA"], "cities": []},
                              "medical": {"departments": [], "specialties": [], "capabilities": []},
                              "joins_required": [], "metrics": {}, "assumptions": []})
        elif "SQL Agent" in prompt:
            out = self.sql
        elif "healthcare data analyst" in prompt:
            q = re.search(r"User question: (.*)", prompt)
            out = f"ANSWER TO: {q.group(1).strip() if q else '?'}"
        else:
            out = "{}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])


class HashEmbedding:
    """Chroma embedding function that needs no network."""

    def __call__(self, input):
        import hashlib
        out = []
        for text in input:
            v = [0.0] * 64
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
            out.append(v)
        return out

    def name(self):  # chromadb >= 0.5 asks embedding functions for a name
        return "hash-test"


@pytest.fixture
def llm():
    return ScriptedLLM(prompts=[])


@pytest.fixture
def graph_env(monkeypatch, tmp_path, llm):
    """Point the app at a temp DB and the scripted LLM, and reset the
    cached graph so each test builds its own."""
    from medical_deserts import graph as graph_mod
    from medical_deserts.config import Config

    monkeypatch.setattr(Config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(Config, "get_llm", classmethod(lambda cls: llm))
    monkeypatch.setattr(graph_mod, "_compiled_graph", None)

    from medical_deserts.retrieval import vector

    monkeypatch.setattr(vector.VectorAgent, "_get_embedding_function", lambda self: HashEmbedding(), raising=False)
    return graph_mod
