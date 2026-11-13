"""Bounded PEDRA workflow: plan, retrieve, execute, answer, review, revise once."""
from __future__ import annotations
import json
from .planner import make_plan
from .evidence_ledger import EvidenceLedger
from .critic import review, semantic_review, decide_review
from .answer import create_answer, evidence_only_response, SYSTEM
from app.core.llm_client import LLMClient, LLMUnavailable
from app.tools import ToolRegistry

DISCLAIMER = 'This is a corpus-grounded technical demonstration, not medical advice. Consult a qualified clinician and official labeling.'

class PEDRA:
    def __init__(self, index, llm=None):
        self.index = index
        self.llm = llm if llm is not None else LLMClient()
        self.tools = ToolRegistry(index)

    def answer(self, question: str) -> dict:
        known = sorted({c['medication'] for c in self.index.chunks if c['medication']})
        plan = make_plan(question, known, self.llm)
        ledger = EvidenceLedger()
        trace = []
        retrieved = self.tools.retrieve_evidence(question, plan.medications, plan.sections)
        ledger.add(retrieved)
        trace.append({'tool': 'retrieve_evidence', 'arguments': {'question': question, 'medications': plan.medications,
            'sections': plan.sections}, 'status': 'ok', 'evidence_ids': [e['id'] for e in retrieved]})
        initial = review(question, ledger, plan.medications)
        if initial['missing_medications']:
            for drug in initial['missing_medications']:
                additional = self.index.search(question + ' ' + drug, medication=drug, top_k=5)
                ledger.add(additional)
                trace.append({'tool': 'retrieve_evidence', 'arguments': {'medication': drug, 'retry': True},
                              'status': 'ok' if additional else 'no_results', 'evidence_ids': [e['id'] for e in additional]})

        results = {}
        executed = ['retrieve_evidence']
        for name in plan.tools:
            if name == 'retrieve_evidence':
                continue
            if name == 'compare_evidence':
                outcome = self.tools.compare_evidence(ledger.items, plan.medications)
                args = {'medications': plan.medications}
            elif name == 'evaluate_threshold':
                outcome = self.tools.evaluate_threshold(ledger.items, plan.medications, question)
                args = {'question': question, 'medications': plan.medications}
            else:
                continue
            results[name] = outcome
            trace.append({'tool': name, 'arguments': args, 'status': outcome['status'], 'result': outcome})
            if outcome['status'] != 'skipped':
                executed.append(name)

        extra = '\n\nLOCAL TOOL RESULTS (not a substitute for the original sources):\n' + json.dumps(results, ensure_ascii=False)
        try:
            answer = create_answer(question + extra, ledger, self.llm)
        except LLMUnavailable:
            answer = ('The configured LLM is unavailable; no clinical synthesis was generated. ' +
                      ' '.join(f"[{e['id']}] {e['source']} — {e['section']}: {e['text'][:220]}" for e in ledger.items[:4]))

        critic = review(question, ledger, plan.medications, answer)
        semantic = semantic_review(question, ledger, answer, self.llm) if ledger.items else {
            'checked': False, 'passed': None, 'unsupported_claims': [], 'ambiguous_claims': [],
            'material_ambiguities': [], 'reason': 'No source evidence'}
        critic['semantic_review'] = semantic
        unit_warning = bool(results.get('evaluate_threshold', {}).get('unit_warning'))
        decision = decide_review(critic, semantic, answer, unit_warning=unit_warning)
        repair_needed = decision['status'] == 'INSUFFICIENT_EVIDENCE' or (decision['status'] == 'CONDITIONAL' and self.llm.enabled)
        revision_attempted = False
        answer_withheld = False
        if repair_needed and self.llm.enabled and ledger.items:
            revision_attempted = True
            try:
                sources = '\n\n'.join(f"[{e['id']}] {e['source']} / {e['section']}\n{e['text']}" for e in ledger.items[:12])
                revised = self.llm.complete(SYSTEM,
                    f'Question: {question}\nPrevious answer: {answer}\nAudit: {json.dumps(semantic)}\n'
                    f'Citation problems: {critic["invalid_citations"]}\n'
                    f'Use ONLY the evidence below. Repair unsupported claims and material ambiguities, '
                    f'and remove any comparative ranking or causal statement that is not directly supported. '
                    f'Keep pharmacokinetic exposure effects separate from pharmacodynamic bleeding risk. '
                    f'If clinical units are uncertain, START your answer with an IF conditional, '
                    f'and never start with an unqualified Yes/No. State the source threshold and the '
                    f'unresolved unit issue explicitly; do not invent a conversion. '
                    f'Do not add new unsupported statements.\n{sources}')
                revised_check = review(question, ledger, plan.medications, revised)
                revised_semantic = semantic_review(question, ledger, revised, self.llm)
                revised_decision = decide_review(revised_check, revised_semantic, revised, unit_warning=unit_warning)
                if revised_decision['status'] in ('PASS', 'CONDITIONAL'):
                    answer, critic = revised, revised_check
                    critic['semantic_review'] = revised_semantic
                    trace.append({'tool': 'critic_revision', 'status': 'accepted'})
                else:
                    trace.append({'tool': 'critic_revision', 'status': 'rejected',
                                  'reason': 'Revised answer failed citation or semantic check'})
                    answer = evidence_only_response(ledger)
                    answer_withheld = True
                    trace.append({'tool': 'safe_evidence_fallback', 'status': 'used',
                                  'reason': 'The draft and bounded revision were not sufficiently grounded'})
            except LLMUnavailable:
                trace.append({'tool': 'critic_revision', 'status': 'error', 'reason': 'LLM unavailable'})
        final_decision = decide_review(critic, critic['semantic_review'], answer, unit_warning=unit_warning)
        if answer_withheld:
            final_decision = {
                'status': 'INSUFFICIENT_EVIDENCE',
                'passed': False,
                'reason': 'The generated synthesis was withheld after the bounded revision did not pass evidence review.'
            }
        critic['status'] = final_decision['status']
        critic['passed'] = final_decision['passed']
        critic['decision_reason'] = final_decision['reason']
        critic['revision_attempted'] = revision_attempted
        critic['answer_withheld'] = answer_withheld
        critic['claim_links'] = ledger.link_answer(answer)
        if critic['status'] == 'INSUFFICIENT_EVIDENCE' and not answer_withheld:
            answer += '\n\nWARNING: Evidence or clinical-claim verification is incomplete. Do not rely on this answer.'
        elif critic['status'] == 'CONDITIONAL':
            answer += '\n\nNOTE: This conclusion is conditional on confirmation of the reported measurement units.'
        return {'question': question, 'answer': answer, 'citations': ledger.citations(),
                'plan': plan.subquestions, 'tools_used': executed, 'tool_results': results,
                'tool_trace': trace, 'critic': critic, 'disclaimer': DISCLAIMER,
                'planning_mode': plan.strategy}
