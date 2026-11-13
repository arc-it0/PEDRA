"""LLM-driven plan; explainable keyword fallback if the LLM is not configured."""
from dataclasses import dataclass
import re
from app.core.llm_client import LLMUnavailable

SECTION_HINTS = {
    "contraindicat": ["contraindications"],
    "interaction": ["drug interactions"],
    "bleed": ["warning", "drug interactions"],
    "dose": ["dosage"],
    "egfr": ["renal", "contraindications", "dosage"],
    "renal": ["renal", "contraindications", "dosage"],
    "kidney": ["renal", "contraindications", "dosage"],
}
TOOLS = {"retrieve_evidence", "compare_evidence", "evaluate_threshold"}

@dataclass
class Plan:
    subquestions: list[str]
    medications: list[str]
    sections: list[str]
    tools: list[str]
    strategy: str = "rules"

def numeric_units(text: str) -> list[str]:
    """Capture user-specified renal numeric expressions without guessing normalization."""
    pattern = r"\b\d+(?:[.,]\d+)?\s*(?:mL\s*/\s*min(?:ute)?(?:\s*/\s*1[.,]73\s*m(?:\^?2|²))?)"
    return [re.sub(r"\s+", "", x).lower() for x in re.findall(pattern, text, flags=re.IGNORECASE)]


def preserve_numeric_question(question: str, steps: list[str]) -> list[str]:
    """Reject an LLM plan that silently changes the numeric expressions in the original question."""
    original = numeric_units(question)
    if not original:
        return steps
    # Every numeric unit in a plan must have appeared in the question, verbatim after whitespace normalization.
    if any(unit not in original for step in steps for unit in numeric_units(step)):
        return [question, "Check whether the reported units match the source-label units without assuming any conversion."]
    # Keep the original measurement visible to downstream reviewers, even when omitted by the planner.
    if not all(unit in numeric_units(" ".join(steps)) for unit in original):
        return [question] + steps[:5]
    return steps


def make_plan(question: str, known_medications: list[str], llm=None) -> Plan:
    low = question.lower()
    named = [m for m in known_medications if re.search(rf"\b{re.escape(m.lower())}\b", low)]
    sections = list(dict.fromkeys(v for key, values in SECTION_HINTS.items() if key in low for v in values))
    tools = ["retrieve_evidence"]
    if len(named) >= 2 or any(w in low for w in ("compare", "versus", " vs ", "difference")):
        tools.append("compare_evidence")
    if any(w in low for w in ("egfr", "threshold", "renal", "kidney", "creatinine")):
        tools.append("evaluate_threshold")
    fallback = Plan([question], named, sections, tools)
    if llm is None or not llm.enabled:
        return fallback
    try:
        proposal = llm.json(
            "You are a medical-document research planner, not a clinician. Choose tools based on informational needs, not fixed question patterns. "
            "Preserve every numerical value and its units exactly as provided by the user in subquestions. Never normalize, convert, infer, or add missing units. Explicitly ask to verify uncertain unit normalization. "
            "Never invent drug names. Available tools: retrieve_evidence, compare_evidence, evaluate_threshold. "
            "Return keys subquestions (strings), medications (strings), sections (strings), tools (strings). "
            "Always include retrieve_evidence. compare_evidence compares documents; evaluate_threshold extracts numeric conditions and is ONLY appropriate for an explicit numerical threshold or cut-off question. Do not select it just because the question mentions INR monitoring or drug interactions.",
            f"Corpus medication names: {known_medications}\nQuestion: {question}",
        )
        meds = [m for m in known_medications if m.lower() in {str(x).lower() for x in proposal.get("medications", [])}]
        # Always retain explicit medication mentions even if a model overlooks one.
        meds = list(dict.fromkeys(named + meds))
        steps = [str(x)[:300] for x in proposal.get("subquestions", []) if isinstance(x, str)][:6] or fallback.subquestions
        steps = preserve_numeric_question(question, steps)
        sects = [str(x)[:100] for x in proposal.get("sections", []) if isinstance(x, str)][:8]
        choices = [str(x) for x in proposal.get("tools", []) if str(x) in TOOLS]
        return Plan(steps, meds, sects or sections, list(dict.fromkeys(["retrieve_evidence"] + choices)), "llm")
    except (ValueError, LLMUnavailable, TypeError, KeyError):
        return fallback
