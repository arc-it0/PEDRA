from app.agent.critic import decide_review

CITES = {"passed": True}

def audit(passed=True, material=None, unsupported=None):
    return {"passed": passed, "material_ambiguities": material or [], "unsupported_claims": unsupported or []}


def test_clean_claim_passes():
    result = decide_review(CITES, audit(), "Supported fact [c1].")
    assert result["status"] == "PASS" and result["passed"]


def test_unqualified_yes_with_unit_difference_fails():
    result = decide_review(CITES, audit(False, ["Unknown normalized units"]),
                           "Yes, eGFR 25 mL/min means contraindicated. Units need verification.", unit_warning=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"


def test_explicit_condition_can_be_returned_but_not_passing():
    result = decide_review(CITES, audit(False, ["Unknown normalized units"]),
                           "If the eGFR of 25 is in mL/min/1.73 m², it falls below the labeled threshold [c1].", unit_warning=True)
    assert result["status"] == "CONDITIONAL" and result["passed"] is False


def test_unsupported_claim_is_not_accepted_even_if_conditional():
    result = decide_review(CITES, audit(False, unsupported=["Invented dosage"]),
                           "If eGFR is normalized, take double dosage [c1].", unit_warning=True)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"


def test_semantic_audit_unavailable_never_passes():
    result = decide_review(CITES, {"passed": None}, "Supported fact [c1].")
    assert result["status"] == "INSUFFICIENT_EVIDENCE"


def test_equal_normalized_units_are_applicable(tmp_path):
    from app.tools.registry import ToolRegistry
    from app.retrieval.search import EvidenceIndex
    evidence = [{"id": "e1", "medication": "Metformin", "text": "eGFR below 30 mL/min/1.73 m2 is contraindicated."}]
    (tmp_path / "dailymed-metformin.md").write_text("# RENAL\n\neGFR below 30 mL/min/1.73 m2 is contraindicated.", encoding="utf-8")
    result = ToolRegistry(EvidenceIndex(tmp_path)).evaluate_threshold(
        evidence, ["Metformin"], "If eGFR is 25 mL/min/1.73 m2, is it below 30?")
    assert result["status"] == "ok"
    assert result["matches"][0]["clinically_applicable"] is True


def test_agent_accepts_conditional_revision_when_units_are_uncertain(tmp_path, monkeypatch):
    from app.agent import PEDRA
    from app.retrieval.search import EvidenceIndex
    from app.agent import orchestrator as orch
    (tmp_path / 'dailymed-metformin.md').write_text(
        '# 4 CONTRAINDICATIONS\n\nMetformin is contraindicated when eGFR is below 30 mL/min/1.73 m2.', encoding='utf-8')

    class FakeLLM:
        enabled = True
        def json(self, system, prompt):
            return {"subquestions": ["Identify the renal threshold"], "medications": ["Metformin"],
                    "sections": ["Contraindications"], "tools": ["retrieve_evidence", "evaluate_threshold"]}
        def complete(self, system, prompt, max_tokens=1300):
            if 'Previous answer:' in prompt:
                return 'If eGFR 25 is expressed in mL/min/1.73 m², it is below 30, the label contraindication threshold [dailymed-metformin-TEST]. Confirm normalized units.'
            return 'Yes. Metformin is contraindicated [dailymed-metformin-TEST].'

    # Fixed evidence IDs make the fake completion genuinely verifiable.
    index = EvidenceIndex(tmp_path)
    real_id = index.chunks[0]['id']
    fake = FakeLLM()
    real_complete = fake.complete
    fake.complete = lambda system, prompt, max_tokens=1300: real_complete(system, prompt, max_tokens).replace('dailymed-metformin-TEST', real_id)
    monkeypatch.setattr(orch, 'semantic_review', lambda question, ledger, answer, llm: {
        'checked': True, 'passed': False, 'unsupported_claims': [], 'ambiguous_claims': [],
        'material_ambiguities': ['The input eGFR unit normalization is unknown'],
        'reason': 'Normalization not established'})
    response = PEDRA(index, llm=fake).answer('At eGFR 25 mL/min, is metformin contraindicated?')
    assert response['critic']['status'] == 'CONDITIONAL'
    assert response['critic']['passed'] is False
    assert any(t['tool'] == 'critic_revision' and t['status'] == 'accepted' for t in response['tool_trace'])


def test_agent_withholds_a_rejected_draft_instead_of_displaying_it(tmp_path, monkeypatch):
    from app.agent import PEDRA
    from app.retrieval.search import EvidenceIndex
    from app.agent import orchestrator as orch
    (tmp_path / 'dailymed-warfarin.md').write_text(
        '# DRUG INTERACTIONS\n\nWarfarin exposure can change with concomitant medicines.', encoding='utf-8')
    index = EvidenceIndex(tmp_path)
    evidence_id = index.chunks[0]['id']

    class FakeLLM:
        enabled = True
        def json(self, system, prompt):
            return {"subquestions": ["Review interactions"], "medications": ["Warfarin"],
                    "sections": ["DRUG INTERACTIONS"], "tools": ["retrieve_evidence"]}
        def complete(self, system, prompt, max_tokens=1300):
            return f'Warfarin has a broader interaction profile [{evidence_id}]'

    monkeypatch.setattr(orch, 'semantic_review', lambda *args, **kwargs: {
        'checked': True, 'passed': False, 'unsupported_claims': ['Broader profile is not established'],
        'ambiguous_claims': [], 'material_ambiguities': [], 'reason': 'Unsupported comparison'})
    response = PEDRA(index, llm=FakeLLM()).answer('Review warfarin interactions')
    assert response['critic']['answer_withheld'] is True
    assert response['critic']['status'] == 'INSUFFICIENT_EVIDENCE'
    assert 'broader interaction profile' not in response['answer']
    assert 'withheld the generated clinical synthesis' in response['answer']
    assert any(t['tool'] == 'safe_evidence_fallback' and t['status'] == 'used' for t in response['tool_trace'])
