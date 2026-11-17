"""第5回の教材と各モードの統合確認。外部APIは呼ばない。"""

import json
import struct
import sys

import pytest
from pydantic_ai.models.test import TestModel

from app import agent as agent_module
from app import retrieval


@pytest.mark.parametrize(
    ("mode", "names"),
    [
        ([], {"search_materials"}),
        (["--tree"], {"browse_tree"}),
        (["--graph"], {"search_graph"}),
        (["--full-context"], set()),
        (["--glossary"], {"search_glossary", "get_lecture_summary", "make_quiz"}),
    ],
)
def test_cli_offers_only_the_selected_mode_tools(mode, names, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["agent", *mode, "--list-tools"])
    agent_module.main()
    lines = capsys.readouterr().out.splitlines()
    assert {line[1:-1] for line in lines if line.startswith("[")} == names


def test_glossary_cli_keeps_the_previous_tools(monkeypatch, capsys):
    model = TestModel()
    monkeypatch.setattr(agent_module, "build_model", lambda: model)
    monkeypatch.setattr(sys, "argv", ["agent", "--glossary", "--trace", "第2回の確認問題"])
    agent_module.main()
    output = capsys.readouterr().out
    for name in ("search_glossary", "get_lecture_summary", "make_quiz"):
        assert f"[tool call]   {name}(" in output
    assert "[usage]" in output
    assert "[cost]" in output


def test_bundled_index_matches_materials_and_supports_search(monkeypatch):
    pages = {page["page_id"]: page for page in retrieval.load_pages()}
    with retrieval.open_index(retrieval.RAG_DB) as db:
        rows = db.execute("SELECT page_id, content FROM chunks").fetchall()
        assert {row["page_id"]: row["content"] for row in rows} == {
            page_id: page["content"] for page_id, page in pages.items()
        }
        blob = db.execute(
            "SELECT embedding FROM chunks WHERE page_id = ?", ("lec02-main-p052",)
        ).fetchone()[0]
    vector = struct.unpack(f"<{len(blob) // 4}f", blob)
    assert len(vector) == retrieval.EMBEDDING_DIMENSIONS
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: list(vector))
    hits = retrieval.search("LoRA QLoRA")
    assert len(hits) == retrieval.RAG_TOP_K
    assert any(hit["page_id"] == "lec02-main-p052" for hit in hits)
    assert hits[0]["distance"] == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize("inside_section", [False, True])
def test_tree_can_reach_page_text(inside_section):
    root = retrieval.browse_tree()
    assert {node["id"] for node in root["children"]} == set(retrieval.RAG_DOCUMENT_IDS)
    lecture = retrieval.browse_tree("lec02-main")
    parent = lecture
    if inside_section:
        section = next(node for node in lecture["children"] if node["kind"] == "section")
        parent = retrieval.browse_tree(section["id"])
    preview = next(node for node in parent["children"] if node["kind"] == "page")
    page = retrieval.browse_tree(preview["id"])
    assert page["children"] == []
    assert page["node"]["content"] == retrieval.read_page(preview["id"])["content"]
    assert page["node"]["content"].startswith(preview["content"])


def test_graph_expands_from_a_retrieved_page(monkeypatch):
    page = retrieval.read_page("lec02-main-p052")
    monkeypatch.setattr(retrieval, "search", lambda _: [{**page, "distance": 0.0}])
    result = retrieval.search_graph("LoRA QLoRA")
    assert result["seeds"][0]["page_id"] == page["page_id"]
    assert result["expanded"]
    candidates = result["seeds"] + result["expanded"]
    assert len(candidates) <= retrieval.GRAPH_MAX_RESULTS
    assert len({hit["page_id"] for hit in candidates}) == len(candidates)
    for hit in result["expanded"]:
        assert hit["content"] == retrieval.read_page(hit["page_id"])["content"]
        assert 1 <= hit["hops"] <= retrieval.GRAPH_HOPS


def test_full_context_includes_all_documents_without_tools():
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    model = TestModel()
    result = retrieval.run_full_context(
        agent_module.agent, "LoRAとは？", model,
        instructions=agent_module.FULL_CONTEXT_INSTRUCTIONS,
        prompt_template=agent_module.FULL_CONTEXT_PROMPT,
    )
    assert model.last_model_request_parameters.function_tools == []
    prompt = next(
        part.content
        for message in result.all_messages() if isinstance(message, ModelRequest)
        for part in message.parts if isinstance(part, UserPromptPart)
    )
    documents = json.loads(retrieval.RAG_SOURCE.read_text())
    for document in documents:
        assert f"資料: {document['id']} / {document['title']}" in prompt
    assert "質問: LoRAとは？" in prompt
