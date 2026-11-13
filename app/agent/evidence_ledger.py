"""Trace claims to exact, full corpus chunks; never store invented evidence."""
from dataclasses import dataclass, field
import re

@dataclass
class EvidenceLedger:
    items: list[dict] = field(default_factory=list)
    claim_links: list[dict] = field(default_factory=list)

    def add(self, evidence: list[dict]) -> None:
        known = {item['id'] for item in self.items}
        for item in evidence:
            if item['id'] not in known:
                self.items.append(item)
                known.add(item['id'])

    def link_answer(self, answer: str) -> list[dict]:
        """Extract explicit claim-to-chunk references; semantic support is checked separately."""
        known = {x['id'] for x in self.items}
        links = []
        for line in answer.splitlines():
            line = line.strip().lstrip('-* ').strip()
            if not line:
                continue
            ids = list(dict.fromkeys(re.findall(r'\[([A-Za-z0-9_-]+)\]', line)))
            if ids:
                links.append({'claim': re.sub(r'\[[A-Za-z0-9_-]+\]', '', line).strip(),
                              'evidence_ids': [x for x in ids if x in known]})
        self.claim_links = links
        return links

    def citations(self) -> list[dict]:
        return [{'id': item['id'], 'source': item['source'], 'medication': item.get('medication'),
                 'section': item['section'], 'excerpt': item['text'][:420], 'passage': item['text'],
                 'score': item['score']}
                for item in self.items]
