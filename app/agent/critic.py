"""Citation/coverage gates plus optional LLM-based claim-level semantic audit."""
import re

def review(question: str, ledger, medications: list[str], answer: str = '') -> dict:
    present = {(item.get('medication') or '').lower() for item in ledger.items}
    missing = [m for m in medications if m.lower() not in present]
    known = {x['id'] for x in ledger.items}
    cited = re.findall(r'\[([A-Za-z0-9_-]+)\]', answer)
    invalid = sorted({cid for cid in cited if cid not in known})
    passed = bool(ledger.items) and not missing and not invalid and (not answer or bool(cited))
    return {'passed': passed, 'evidence_count': len(ledger.items), 'missing_medications': missing,
            'invalid_citations': invalid, 'cited_chunk_count': len(set(cited)),
            'reason': 'Citation IDs and medicine coverage verified; clinical support requires semantic review.' if passed else 'Evidence, medicine coverage, or citations incomplete.'}

def semantic_review(question: str, ledger, answer: str, llm) -> dict:
    if not llm.enabled:
        return {'checked': False, 'passed': None, 'unsupported_claims': [], 'ambiguous_claims': [], 'reason': 'No LLM configured'}
    from app.core.llm_client import LLMUnavailable
    context = '\n\n'.join(f"[{e['id']}] SOURCE: {e['source']} SECTION: {e['section']}\n{e['text']}" for e in ledger.items[:12])
    try:
        result = llm.json(
            'You audit a pharmacology research answer using ONLY the provided sources. Check each distinct clinical claim and unit. '
            'A claim is unsupported if its cited text does not support it. An ambiguity is MATERIAL if it can change medical interpretation, '
            'including risk vs contraindication and mL/min vs mL/min/1.73m². '
            'Return JSON fields: passed:boolean, unsupported_claims:list of strings, ambiguous_claims:list of strings, '
            'material_ambiguities:list of strings, reason:string. Set passed=false if any unsupported claim or material ambiguity. '
            'When an answer clearly states an unknown unit assumption, still report the unit uncertainty as material; '
            'a transparent qualification does not make the underlying clinical measurement certain.',
            f'Question: {question}\nAnswer: {answer}\nEvidence:\n{context}')
        unsupported = [str(x) for x in result.get('unsupported_claims', []) if isinstance(x, str)][:8]
        ambiguous = [str(x) for x in result.get('ambiguous_claims', []) if isinstance(x, str)][:8]
        material = [str(x) for x in result.get('material_ambiguities', []) if isinstance(x, str)][:8]
        return {'checked': True, 'passed': result.get('passed') is True and not unsupported and not material,
                'unsupported_claims': unsupported, 'ambiguous_claims': ambiguous,
                'material_ambiguities': material, 'reason': str(result.get('reason', ''))[:500]}
    except (LLMUnavailable, ValueError, TypeError, KeyError) as exc:
        return {'checked': False, 'passed': None, 'unsupported_claims': [], 'ambiguous_claims': [],
                'material_ambiguities': [], 'reason': f'Audit unavailable: {type(exc).__name__}'}


def decide_review(citation_review: dict, semantic: dict, answer: str, *, unit_warning: bool = False) -> dict:
    """Combine all checks into a consistent 3-state decision.

    PASS: independently supported answer, no material uncertainty.
    CONDITIONAL: supported factual answer, but clinically material context is
                 unresolved and explicitly disclosed, without a direct
                 unqualified clinical decision.
    INSUFFICIENT_EVIDENCE: failed grounding or undisclosed material uncertainty.
    """
    if not citation_review['passed'] or semantic.get('passed') is None:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'passed': False, 'reason': 'Citation verification failed or semantic audit unavailable.'}
    unsupported = semantic.get('unsupported_claims', [])
    material = semantic.get('material_ambiguities', [])
    if unsupported:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'passed': False, 'reason': 'Unsupported clinical claims detected.'}
    if not material and not unit_warning and semantic.get('passed') is True:
        return {'status': 'PASS', 'passed': True, 'reason': 'Citations and clinical claims verified.'}
    # Qualifying unknown units means stating the condition BEFORE the conclusion.
    # Merely appending a caution after an unqualified Yes is not enough.
    initial = answer.lstrip().split('\n', 1)[0].lower()
    has_conditional = bool(re.search(r'\b(if|assuming|provided that|only if|cannot determine|not possible to determine)\b', initial))
    discloses_units = bool(re.search(r'(?:unit|normaliz|1[.,]73\s*m|body.surface)', answer, re.I))
    if material and not unit_warning:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'passed': False, 'reason': 'Material clinical ambiguity remains unresolved.'}
    if unit_warning and has_conditional and discloses_units:
        return {'status': 'CONDITIONAL', 'passed': False, 'reason': 'Clinical threshold requires confirmation of eGFR units; answer is explicitly conditional.'}
    return {'status': 'INSUFFICIENT_EVIDENCE', 'passed': False, 'reason': 'Material ambiguity is not adequately qualified before the decision.'}
