"""Source-bound answer generation and a transparent no-LLM fallback."""
from app.core.llm_client import LLMUnavailable

SYSTEM = """You are PEDRA, a careful document-grounded pharmacology research assistant.
Use ONLY the supplied source snippets. Answer the question directly; for comparisons, discuss each drug.
Every factual clinical statement must cite its evidence in the exact format [chunk-id].
Never cite unseen chunks. Distinguish contraindications, warnings and precautions precisely.
If evidence is absent or ambiguous, state that limitation explicitly; do not add outside medical knowledge.
Do not claim that one medicine has more, broader, or fewer interactions overall unless a supplied source directly makes that comparison.
Distinguish pharmacokinetic exposure effects from pharmacodynamic clinical risks. In particular, an absence of a significant
pharmacokinetic effect does not establish an absence of bleeding risk with medicines that affect hemostasis.
Do not describe CYP3A4/P-gp effects as "combined" or "principal" unless the supplied evidence uses that precise scope.
Do not make individualized treatment decisions. Conclude before the separate disclaimer.
"""

def evidence_only_response(ledger) -> str:
    """Fail closed when an answer draft cannot be grounded after bounded revision."""
    if not ledger.items:
        return "PEDRA could not retrieve sufficient evidence to produce a validated response."
    passages = []
    for item in ledger.items[:8]:
        passages.append(
            f"- **{item['source']} — {item['section']}** [{item['id']}]\n"
            f"  {item['text'].strip()}"
        )
    return (
        "PEDRA withheld the generated clinical synthesis because its evidence review found unsupported or "
        "materially ambiguous claims. The retrieved source passages below are provided for traceability; "
        "they do not establish an individualized clinical conclusion.\n\n"
        "## Retrieved source passages\n" + "\n\n".join(passages)
    )

def create_answer(question, ledger, llm):
    if not ledger.items:
        return "The local corpus did not provide sufficient relevant evidence to answer this question."
    snippets = "\n\n".join(
        f"[{e['id']}] FILE: {e['source']} | SECTION: {e['section']}\n{e['text']}"
        for e in ledger.items[:12]
    )
    if llm.enabled:
        return llm.complete(SYSTEM, f"Question: {question}\n\nEVIDENCE:\n{snippets}", max_tokens=1300)
    return ("LLM is not configured. The following passages were retrieved; no clinical synthesis has "
            "been generated. " + " ".join(f"[{e['id']}] {e['source']} — {e['section']}: {e['text'][:220]}" for e in ledger.items[:4]))
