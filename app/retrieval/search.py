from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .chunker import chunk_document
from .embeddings import LiteLLMEmbedder

class EvidenceIndex:
    def __init__(self, corpus_dir: Path):
        self.chunks = [chunk for path in sorted(corpus_dir.glob("*.md")) for chunk in chunk_document(path)]
        if not self.chunks:
            raise ValueError("No Markdown corpus documents found")
        texts = [f"{c['medication'] or ''} {c['section']} {c['text']}" for c in self.chunks]
        self.embedder = LiteLLMEmbedder()
        self.vectors = self.embedder.embed(texts)
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=25000)
        self.tfidf_vectors = self.tfidf.fit_transform(texts)

    def search(self, query: str, top_k: int = 6, medication: str | None = None,
               sections: list[str] | None = None, baseline: bool = False) -> list[dict]:
        allowed = np.ones(len(self.chunks), dtype=bool)
        if medication:
            allowed &= np.array([(c["medication"] or "").lower() == medication.lower() for c in self.chunks])
        if baseline:
            scores = (self.tfidf.transform([query]) @ self.tfidf_vectors.T).toarray().ravel()
        else:
            scores = (self.embedder.embed([query]) @ self.vectors.T).ravel()
        # Prefer a requested section but do not exclude relevant neighboring sections.
        if sections:
            targets = [s.lower() for s in sections]
            scores += np.array([0.12 if any(s in c["section"].lower() for s in targets) else 0.0 for c in self.chunks])
        scores[~allowed] = -np.inf
        chosen = [int(i) for i in np.argsort(scores)[::-1] if np.isfinite(scores[i]) and scores[i] > 0][:top_k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in chosen]
