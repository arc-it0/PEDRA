# PEDRA — Pharma Evidence-Driven Research Agent

PEDRA is an evidence-driven pharmaceutical research agent. It analyses a user question, retrieves relevant passages from a local pharmaceutical corpus, produces a cited answer, and runs consistency checks before presenting the result.

> **Disclaimer:** PEDRA is a technical demonstration based on a local corpus. It is not medical advice, a diagnosis, or a treatment recommendation. Clinical decisions must be verified against official sources and reviewed by a qualified healthcare professional.

---

## Features

- Professional SYNTAX-branded web interface
- Natural-language search over a local Markdown pharmaceutical corpus
- Source-cited answers with supporting excerpts
- Full source-passage inspection from the interface
- Research plan and executed-tool traceability
- Citation, medicine-coverage, and clinical-ambiguity checks
- Evidence-review statuses:
  - `PASS` — evidence checks passed
  - `CONDITIONAL` — clarification is required before a definitive conclusion
  - `INSUFFICIENT_EVIDENCE` — the available evidence is not sufficient for a reliable synthesis
- Browser-local research notes
- FastAPI API with interactive documentation

---

## Architecture

```text
User question
     |
     v
Research planning
     |
     v
Local corpus retrieval
     |
     v
Specialized tools
(retrieve_evidence, compare_evidence, evaluate_threshold)
     |
     v
Evidence ledger and stable citations
     |
     v
Answer synthesis
     |
     v
Evidence review and consistency checks
     |
     v
PEDRA cited response
```

The local corpus contains pharmaceutical monographs in Markdown format. Documents are split into sections and passages, then indexed for targeted retrieval.

---

## Technology Stack

- Python 3.11+
- FastAPI
- Uvicorn
- OpenAI SDK, compatible with a configured API/proxy
- Scikit-learn and NumPy for document retrieval
- Native HTML, CSS, and JavaScript frontend
- Pytest for automated tests

---

## Project Structure

```text
PEDRA/
├── app/
│   ├── agent/              # Planning, orchestration, answer generation, critic
│   ├── core/               # Configuration and LLM client
│   ├── retrieval/          # Indexing, embeddings, and search
│   ├── tools/              # Specialized research tools
│   ├── main.py             # FastAPI application
│   └── schemas.py          # Request and response models
├── corpus/                 # Local Markdown pharmaceutical sources
├── frontend/
│   └── index.html          # PEDRA / SYNTAX web interface
├── tests/                  # Automated tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md
```

---

## Local Installation (Windows PowerShell)

### 1. Open the project folder

```powershell
cd D:\Syntaxe\PEDRA
```

### 2. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

If it does not exist yet, create it first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create the `.env` configuration file

Create a file called `.env` in the project root with the following values:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_api_base_url_here
OPENAI_MODEL=your_model_name_here
OPENAI_EMBEDDING_MODEL=your_embedding_model_name_here
```

> Never share, commit, or upload the `.env` file or API credentials.

### 5. Start PEDRA

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- PEDRA interface: [http://localhost:8000](http://localhost:8000)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health endpoint: [http://localhost:8000/health](http://localhost:8000/health)

Stop the server with:

```powershell
Ctrl + C
```

---

## API Usage

### Check service health

```powershell
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "PEDRA",
  "llm_configured": true
}
```

### Ask PEDRA a research question

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"Compare warfarin and apixaban contraindications"}'
```

Example JSON request:

```json
{
  "question": "A patient with an eGFR of 25 mL/min/1.73 m² is being considered for metformin. Is metformin contraindicated?"
}
```

The response includes:

```json
{
  "question": "...",
  "answer": "...",
  "citations": [],
  "plan": [],
  "tools_used": [],
  "tool_results": {},
  "tool_trace": [],
  "critic": {},
  "disclaimer": "...",
  "planning_mode": "..."
}
```

---

## Run Tests

From the project directory, with `.venv` activated:

```powershell
python -m pytest -q
```

The automated tests cover, among other areas:

- API contracts
- Corpus retrieval
- Citations
- Critic decisions
- Unit-ambiguity cases
- Safe fallback behavior when evidence is insufficient
- Frontend/UI contracts

---

## Docker

Build the image:

```powershell
docker build -t pedra-agent .
```

Run the container with environment variables:

```powershell
docker run --rm -p 8000:8000 --env-file .env pedra-agent
```

Then open:

```text
http://localhost:8000
```

---

## Known Limitations

- The corpus is local, limited, and may become outdated.
- A valid citation confirms that a source is associated with an answer; it does not by itself prove every clinical implication.
- Results must be checked against current official labeling and regulatory documents.
- When data is incomplete, ambiguous, or inconsistent, PEDRA should provide a conditional response or withhold the synthesis.
- PEDRA must not be used as the sole basis for medical decisions.

---


---

## SYNTAX Project

PEDRA was designed as a traceable pharmaceutical-research agent prototype, with a dedicated user interface for exploring evidence, reviewing sources, and producing documented research syntheses.
