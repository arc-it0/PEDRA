from pathlib import Path
from app.agent import PEDRA
from app.agent.evidence_ledger import EvidenceLedger
from app.retrieval.search import EvidenceIndex
from app.tools.registry import ToolRegistry


def test_threshold_result_and_provenance(tmp_path):
    (tmp_path / 'dailymed-metformin.md').write_text(
        '# 4 CONTRAINDICATIONS\n\nMetformin is contraindicated at eGFR below 30 mL/min/1.73 m2.', encoding='utf-8')
    agent = PEDRA(EvidenceIndex(tmp_path))
    response = agent.answer('At an eGFR of 25 mL/min, is metformin contraindicated and what is the threshold?')
    outcome = response['tool_results']['evaluate_threshold']
    assert outcome['status'] == 'ambiguous_units'
    assert outcome['clinical_decision_supported'] is False
    assert all(x['clinically_applicable'] is None for x in outcome['matches'])
    assert any(x['numeric_comparison'] is True and x['threshold'] == 30 for x in outcome['matches'])
    assert outcome['unit_warning']
    assert any(x['tool'] == 'evaluate_threshold' and x['status'] == 'ambiguous_units' for x in response['tool_trace'])


def test_inappropriate_threshold_is_skipped(tmp_path):
    (tmp_path / 'dailymed-warfarin.md').write_text('# DRUG INTERACTIONS\n\nINR monitoring and bleeding risk.', encoding='utf-8')
    registry = ToolRegistry(EvidenceIndex(tmp_path))
    result = registry.evaluate_threshold([], ['Warfarin'], 'What monitoring is recommended with warfarin and sertraline?')
    assert result['status'] == 'skipped'


def test_evidence_ledger_links_answer_claims():
    ledger = EvidenceLedger()
    ledger.add([{'id':'dailymed-drug-123', 'source':'dailymed-drug.md', 'section':'4 CONTRAINDICATIONS',
                 'medication':'Drug', 'text':'Contraindicated in pregnancy.', 'score':0.9}])
    links = ledger.link_answer('Pregnancy is a contraindication [dailymed-drug-123].')
    assert links[0]['evidence_ids'] == ['dailymed-drug-123']


def test_frontend_has_interactive_workspace_javascript():
    html = (Path(__file__).parents[1] / 'frontend' / 'index.html').read_text(encoding='utf-8')
    assert "function showView" in html
    assert "function renderRecent" in html
    assert "function showCitationPassage" in html
