import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tavily import TavilyClient

# ── State ──────────────────────────────────────────────────────────────────────

class Professor(TypedDict):
    name: str
    title: str
    department: str
    university: str
    research_interests: List[str]
    email: str
    profile_url: str
    relevance_summary: str


class AgentState(TypedDict):
    university: str
    research_interests: str
    search_queries: List[str]
    raw_search_results: List[dict]
    professors: List[Professor]
    error: str


# ── Clients ───────────────────────────────────────────────────────────────────

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def get_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in your .env file.")
    return TavilyClient(api_key=api_key)


# ── Node 1: Generate targeted search queries ───────────────────────────────────

def generate_queries(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = f"""
You are a research assistant helping find professors at a specific university.

University: {state["university"]}
Research Interests: {state["research_interests"]}

Generate 4 highly targeted web search queries to find professors at this university
whose research matches these interests. Vary the queries:
- One targeting the university's faculty page/directory
- One targeting specific lab or group pages
- One targeting Google Scholar or ResearchGate profiles
- One more general but specific to the university + topic

Respond with ONLY a JSON array of 4 query strings. No explanation, no markdown.
Example: ["query 1", "query 2", "query 3", "query 4"]
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = result.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw)
        if not isinstance(queries, list):
            raise ValueError
    except Exception:
        # Fallback queries
        uni = state["university"]
        interests = state["research_interests"]
        queries = [
            f"{uni} professor {interests} faculty",
            f"site:{uni.lower().replace(' ', '')}.edu {interests} research",
            f"{uni} {interests} lab research group professor",
            f"{uni} faculty {interests} Google Scholar",
        ]
    return {**state, "search_queries": queries}


# ── Node 2: Execute web searches ───────────────────────────────────────────────

def search_web(state: AgentState) -> AgentState:
    try:
        client = get_tavily()
    except ValueError as e:
        return {**state, "error": str(e), "raw_search_results": []}

    all_results = []
    for query in state["search_queries"]:
        try:
            response = client.search(
                query=query,
                max_results=5,
                search_depth="advanced",
                include_answer=True,
            )
            all_results.append({
                "query": query,
                "answer": response.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:800],
                    }
                    for r in response.get("results", [])
                ],
            })
        except Exception as e:
            all_results.append({"query": query, "error": str(e), "results": []})

    return {**state, "raw_search_results": all_results}


# ── Node 3: Extract structured professor profiles ──────────────────────────────

def extract_professors(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    llm = get_llm()

    # Build a condensed summary of all search results
    search_text = ""
    for item in state["raw_search_results"]:
        search_text += f"\n\n=== Query: {item['query']} ===\n"
        if "answer" in item and item["answer"]:
            search_text += f"Summary: {item['answer']}\n"
        for r in item.get("results", []):
            search_text += f"\nTitle: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n"

    prompt = f"""
You are an academic research assistant. Analyze the search results below and extract
a list of professors from {state["university"]} whose research aligns with:
"{state["research_interests"]}"

Search Results:
{search_text[:6000]}

Extract up to 8 professors. For each, provide:
- name: Full name
- title: Academic title (Professor, Associate Professor, Assistant Professor, etc.)
- department: Department or school name
- university: "{state["university"]}"
- research_interests: List of 3–5 specific research topics (strings)
- email: Email if found, else ""
- profile_url: Best URL to their profile/page, else ""
- relevance_summary: 1–2 sentence explanation of why they match the research interests

Rules:
- Only include professors clearly affiliated with {state["university"]}
- Only include those with genuine overlap with the requested research interests
- If you cannot find enough confirmed professors, return fewer — do NOT hallucinate names
- Respond ONLY with a valid JSON array. No markdown, no explanation.

Format:
[
  {{
    "name": "...",
    "title": "...",
    "department": "...",
    "university": "...",
    "research_interests": ["...", "..."],
    "email": "...",
    "profile_url": "...",
    "relevance_summary": "..."
  }}
]
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = result.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        professors = json.loads(raw)
        if not isinstance(professors, list):
            raise ValueError
    except Exception:
        professors = []

    return {**state, "professors": professors}


# ── Node 4: Rank & deduplicate ─────────────────────────────────────────────────

def rank_results(state: AgentState) -> AgentState:
    if state.get("error") or not state.get("professors"):
        return state

    llm = get_llm()
    prompt = f"""
Given this list of professors, deduplicate (same person may appear twice with slightly
different names) and rank them by relevance to: "{state["research_interests"]}"

Professors JSON:
{json.dumps(state["professors"], indent=2)[:4000]}

Return the deduplicated, ranked list as a valid JSON array in the same format.
No markdown, no explanation — only the JSON array.
"""
    result = llm.invoke([HumanMessage(content=prompt)])
    try:
        raw = result.content.strip().replace("```json", "").replace("```", "").strip()
        ranked = json.loads(raw)
        if not isinstance(ranked, list):
            raise ValueError
    except Exception:
        ranked = state["professors"]

    return {**state, "professors": ranked}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_queries", generate_queries)
    graph.add_node("search_web", search_web)
    graph.add_node("extract_professors", extract_professors)
    graph.add_node("rank_results", rank_results)

    graph.set_entry_point("generate_queries")
    graph.add_edge("generate_queries", "search_web")
    graph.add_edge("search_web", "extract_professors")
    graph.add_edge("extract_professors", "rank_results")
    graph.add_edge("rank_results", END)

    return graph.compile()


support_graph = build_graph()


# ── Public runner ──────────────────────────────────────────────────────────────

def find_professors(university: str, research_interests: str) -> dict:
    initial_state: AgentState = {
        "university": university,
        "research_interests": research_interests,
        "search_queries": [],
        "raw_search_results": [],
        "professors": [],
        "error": "",
    }
    final = support_graph.invoke(initial_state)
    return {
        "university": final["university"],
        "research_interests": final["research_interests"],
        "search_queries": final["search_queries"],
        "professors": final["professors"],
        "error": final.get("error", ""),
        "count": len(final["professors"]),
    }