from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.retrieval import EvidenceIndex
from app.retrieval.embeddings import EmbeddingError
from app.agent import PEDRA
from app.schemas import AskRequest, AskResponse

app = FastAPI(title="PEDRA — Pharma Evidence-Driven Research Agent", version="1.4.0")
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

@lru_cache(maxsize=1)
def get_agent() -> PEDRA:
    return PEDRA(EvidenceIndex(settings.corpus_dir))

@app.get("/health")
def health():
    # A liveness probe must not build the index or make a remote provider call.
    return {"status": "ok", "service": "PEDRA", "llm_configured": bool(settings.api_key and settings.model)}

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        return get_agent().answer(payload.question)
    except (ValueError, EmbeddingError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND)
