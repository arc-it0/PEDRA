from pathlib import Path
import pytest
from pydantic import ValidationError
from app.retrieval.chunker import chunk_document
from app.retrieval.search import EvidenceIndex
from app.agent import PEDRA

def write_doc(folder: Path, name: str, body: str):
    (folder / name).write_text(body, encoding="utf-8")

def test_section_aware_chunk_metadata(tmp_path):
    write_doc(tmp_path, "dailymed-warfarin.md", "# CONTRAINDICATIONS\n\nAvoid in active bleeding.\n\n# DOSAGE\n\nMonitor INR.")
    chunks = chunk_document(tmp_path / "dailymed-warfarin.md")
    assert {c["section"] for c in chunks} == {"CONTRAINDICATIONS", "DOSAGE"}
    assert all(c["medication"] == "Warfarin" and c["id"] for c in chunks)

class DisabledLLM:
    """Offline stub: verify the agent cannot certify claims without a semantic audit."""
    enabled = False


def test_agent_selects_two_tools_and_returns_citations(tmp_path):
    write_doc(tmp_path, "dailymed-warfarin.md", "# DRUG INTERACTIONS\n\nWarfarin interacts with medicines that increase bleeding risk.")
    write_doc(tmp_path, "dailymed-apixaban.md", "# DRUG INTERACTIONS\n\nApixaban has interaction warnings with strong inhibitors.")
    answer = PEDRA(EvidenceIndex(tmp_path), llm=DisabledLLM()).answer("Compare warfarin and apixaban interactions")
    assert len(answer["tools_used"]) >= 2
    assert answer["citations"]
    assert answer["critic"]["status"] == "INSUFFICIENT_EVIDENCE"  # no LLM semantic audit configured

def test_health_and_input_validation():
    from app.main import health
    from app.schemas import AskRequest
    assert health()["status"] == "ok"
    with pytest.raises(ValidationError):
        AskRequest(question="x")
