from pathlib import Path
from app.main import frontend, health
from app.agent.planner import make_plan, numeric_units

class FakeLLM:
    enabled = True
    def json(self, system, prompt):
        return {"subquestions": ["Does eGFR 25 mL/min/1.73 m² contraindicate metformin?"],
                "medications": ["Metformin"], "sections": ["renal"],
                "tools": ["retrieve_evidence", "evaluate_threshold"]}

def test_planner_does_not_invent_indexed_units():
    question = "At eGFR 25 mL/min is metformin contraindicated?"
    plan = make_plan(question, ["Metformin"], FakeLLM())
    assert plan.subquestions[0] == question
    assert "25ml/min/1.73m²" not in numeric_units(" ".join(plan.subquestions))

def test_planner_retains_explicit_indexed_units():
    question = "At eGFR 25 mL/min/1.73 m² is metformin contraindicated?"
    plan = make_plan(question, ["Metformin"], FakeLLM())
    assert any("25ml/min/1.73m²" in numeric_units(step) for step in plan.subquestions)

def test_frontend_has_safe_markdown_and_click_handler():
    html = Path(__file__).resolve().parents[1].joinpath("frontend/index.html").read_text()
    assert "function renderAnswer" in html
    assert "renderAnswer(a,data.answer" in html
    assert "markdown-table" in html
    assert "Open source passage" in html
    assert "addEventListener('click'" in html

def test_http_health_and_frontend():
    assert health()["status"] == "ok"
    page = frontend()
    assert page.path.name == "index.html"
    assert "Pharma Evidence Research Agent" in page.path.read_text(encoding="utf-8")
