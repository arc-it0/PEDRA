"""Local deterministic tools, with explicit arguments, results, and source provenance."""
import re

class ToolRegistry:
    def __init__(self, index):
        self.index = index

    def retrieve_evidence(self, question, medications, sections):
        results = []
        if medications:
            for medication in medications:
                results.extend(self.index.search(question, top_k=5, medication=medication, sections=sections))
        else:
            results = self.index.search(question, top_k=8, sections=sections)
        seen = set()
        return [x for x in results if not (x['id'] in seen or seen.add(x['id']))]

    def compare_evidence(self, evidence, medications):
        if len(medications) < 2:
            return {'status': 'skipped', 'reason': 'Comparison requires at least two named corpus drugs'}
        groups = {m: [{'id': x['id'], 'section': x['section'], 'source': x['source']}
                      for x in evidence if (x.get('medication') or '').lower() == m.lower()]
                  for m in medications}
        return {'status': 'ok' if all(groups.values()) else 'partial', 'by_medication': groups,
                'missing_medications': [m for m, chunks in groups.items() if not chunks]}

    def evaluate_threshold(self, evidence, medications=None, question=''):
        """Extract source thresholds; calculate only when comparator, unit and query value align.

        Avoid inferring clinical eligibility from arbitrary numeric mentions.
        """
        if not re.search(r'\b(?:egfr|threshold|less than|below|above|greater than|\d+\s*(?:mL/min|mg/dL))\b', question, re.I):
            return {'status': 'skipped', 'reason': 'No explicit numerical threshold question'}
        subject = re.search(r'\begfr\s*(?:of|=|is)?\s*(\d+(?:\.\d+)?)', question, re.I)
        qval = float(subject.group(1)) if subject else None
        # The question may omit normalization to 1.73 m²; do not silently
        # convert unindexed mL/min to the indexed unit used in drug labels.
        q_norm = bool(re.search(r'(?:/\s*1[.,]73\s*m\s*(?:2|²)|per\s*1[.,]73\s*m)', question, re.I))
        q_unit = 'mL/min/1.73 m²' if q_norm else ('mL/min (normalization unspecified)' if re.search(r'mL\s*/\s*min', question, re.I) else 'unspecified')
        found = []
        for entry in evidence:
            if medications and (entry.get('medication') or '').lower() not in [m.lower() for m in medications]:
                continue
            content = entry['text']
            for match in re.finditer(r'\begfr\b.{0,95}?\b(below|less than|under|at or below|<=|<|above|greater than|>=|>)\s*(\d+(?:\.\d+)?)\s*(mL\s*/\s*min(?:ute)?(?:\s*/\s*1\s*[.,]?\s*73\s*m\s*(?:2|²))?)?', content, re.I | re.S):
                comparison, raw_threshold, unit = match.group(1), match.group(2), match.group(3)
                op = '<=' if comparison.lower() in ('at or below','<=') else '<' if comparison.lower() in ('below','less than','under','<') else '>=' if comparison.lower() in ('>=',) else '>'
                value = float(raw_threshold)
                result = (qval <= value if op == '<=' else qval < value if op == '<' else qval >= value if op == '>=' else qval > value) if qval is not None else None
                normalized_source = bool(re.search(r'1\s*[.,]\s*73\s*m\s*(?:2|²)', unit or '', re.I))
                units_compatible = bool(qval is not None and unit and (normalized_source == q_norm) and q_unit != 'unspecified')
                found.append({'id': entry['id'], 'threshold': value, 'operator': op,
                              'unit': (unit or 'not specified').strip(), 'question_value': qval,
                              'numeric_comparison': result, 'clinically_applicable': result if units_compatible else None, 'units_compatible': units_compatible, 'question_unit': q_unit, 'context': content[max(0,match.start()-50):match.end()+80].strip()})
        # Preserve compatibility with earlier tests and expose any exact source values.
        if not found:
            numeric = r'(?<!\w)(?:<|>|≤|≥|=)?\s*\d+(?:\.\d+)?\s*(?:mL/min(?:/1\.73\s*m2)?|mg/dL|mg|%)'
            found = [{'id': x['id'], 'values': re.findall(numeric, x['text'], flags=re.I)[:12]}
                     for x in evidence if re.search(numeric, x['text'], flags=re.I)][:12]
            return {'status': 'unresolved', 'matches': found, 'reason': 'No unambiguous eGFR comparator extracted'}
        for entry in found:
            entry['values'] = [f"{entry['threshold']:g} {entry['unit']}"]
        mismatched = any(x['units_compatible'] is False for x in found)
        return {'status': 'ambiguous_units' if mismatched else 'ok', 'matches': found[:12],
                'unit_warning': ('The reported eGFR and source threshold units are not confirmed equivalent; '
                                 'numeric comparisons do not establish clinical applicability.' if mismatched else ''),
                'clinical_decision_supported': not mismatched}
