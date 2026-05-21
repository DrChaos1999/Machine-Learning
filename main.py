from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List

from agent import find_professors

app = FastAPI(
    title="Professor Finder Agent",
    description="LangGraph agent that finds professors by university and research interests",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    university: str = Field(..., min_length=2, max_length=200, description="University name")
    research_interests: str = Field(..., min_length=3, max_length=500, description="Research interests / keywords")


class ProfessorOut(BaseModel):
    name: str
    title: str
    department: str
    university: str
    research_interests: List[str]
    email: str
    profile_url: str
    relevance_summary: str


class SearchResponse(BaseModel):
    university: str
    research_interests: str
    search_queries: List[str]
    professors: List[ProfessorOut]
    count: int
    error: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Find professors at a university matching given research interests.
    Runs a 4-step LangGraph agent: generate queries → web search → extract → rank.
    """
    if not request.university.strip() or not request.research_interests.strip():
        raise HTTPException(status_code=400, detail="University and research interests are required.")
    try:
        result = find_professors(request.university, request.research_interests)
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent": "professor_finder_langgraph"}