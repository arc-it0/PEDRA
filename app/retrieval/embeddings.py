"""Consistent embedding backend: never compare remote corpus vectors against local query vectors."""
import hashlib
import re
import numpy as np
from app.core.config import settings

class EmbeddingError(RuntimeError):
    pass

class LiteLLMEmbedder:
    def __init__(self, dimensions=384, batch_size=32):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.remote = bool(settings.api_key and settings.embedding_model)
        self._client = None

    def _local(self, texts):
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, content in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", content.lower()):
                slot = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
                matrix[row, slot] += 1.0
        return matrix

    def embed(self, texts):
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if not self.remote:
            matrix = self._local(texts)
        else:
            if self._client is None:
                params = {"api_key": settings.api_key, "timeout": 45.0, "max_retries": 1}
                if settings.base_url:
                    params["base_url"] = settings.base_url
                from openai import OpenAI
                self._client = OpenAI(**params)
            parts = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start+self.batch_size]
                try:
                    result = self._client.embeddings.create(model=settings.embedding_model, input=batch)
                    ordered = sorted(result.data, key=lambda x: x.index)
                    if len(ordered) != len(batch):
                        raise EmbeddingError("Incomplete embedding response")
                    parts.extend(row.embedding for row in ordered)
                except Exception as exc:
                    raise EmbeddingError(f"Remote embedding failure: {type(exc).__name__}") from exc
            matrix = np.asarray(parts, dtype=np.float32)
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
