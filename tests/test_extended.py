from pathlib import Path
from app.retrieval.search import EvidenceIndex
from app.agent import PEDRA
from app.agent.critic import review
from app.agent.evidence_ledger import EvidenceLedger
from app.tools.registry import ToolRegistry


def test_threshold_tool_and_traceability(tmp_path):
    (tmp_path / "dailymed-metformin.md").write_text(
        "# 4 CONTRAINDICATIONS\n\nMetformin is contraindicated at eGFR below 30 mL/min.\n", encoding="utf-8")
    agent = PEDRA(EvidenceIndex(tmp_path))
    result = agent.answer("At an eGFR of 25 mL/min, is metformin contraindicated and what is the threshold?")
    assert "evaluate_threshold" in result["tools_used"]
    assert result["tool_results"]["evaluate_threshold"]
    assert any("30 mL/min" in ' '.join(x["values"]) for x in result["tool_results"]["evaluate_threshold"]["matches"])
    assert result["citations"]


def test_invalid_citation_detected():
    ledger = EvidenceLedger()
    ledger.add([{"id": "c1", "medication": "Metformin", "source": "x.md", "section": "X", "text": "source", "score": 1}])
    assert not review("test", ledger, ["Metformin"], "Unsupported claim [no-such-chunk]")["passed"]


def test_missing_medication_coverage(tmp_path):
    (tmp_path / "dailymed-warfarin.md").write_text("# 4 CONTRAINDICATIONS\n\nActive bleeding.\n", encoding="utf-8")
    ledger = EvidenceLedger()
    ledger.add([{"id": "c1", "medication": "Warfarin", "source": "x.md", "section": "X", "text": "source", "score": 1}])
    assert review("compare", ledger, ["Warfarin", "Apixaban"], "Evidence [c1]")["missing_medications"] == ["Apixaban"]


def test_tfidf_baseline_available(tmp_path):
    (tmp_path / "dailymed-warfarin.md").write_text("# CONTRAINDICATIONS\n\nActive bleeding should be considered.\n", encoding="utf-8")
    index = EvidenceIndex(tmp_path)
    assert index.search("active bleeding", baseline=True)[0]["source"] == "dailymed-warfarin.md"
