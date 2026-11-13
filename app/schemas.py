from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

class Citation(BaseModel):
    id: str
    source: str
    medication: str | None = None
    section: str
    excerpt: str
    passage: str
    score: float

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    plan: list[str]
    tools_used: list[str]
    tool_results: dict = Field(default_factory=dict)
    tool_trace: list[dict] = Field(default_factory=list)
    critic: dict
    disclaimer: str
    planning_mode: str
