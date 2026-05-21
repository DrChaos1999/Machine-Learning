# Professor Finder Agent

An AI-powered web app that finds professors at any university matching your research interests. Built with **LangGraph**, **FastAPI**, and **OpenAI GPT-4o-mini**, it runs a 4-step agentic pipeline to search, extract, and rank relevant faculty profiles.

---

## How It Works

```
User Input (university + interests)
        │
        ▼
┌─────────────────────┐
│ Step 1              │  GPT-4o-mini generates 4 targeted
│ Generate Queries    │  search queries (faculty page, labs,
│                     │  Google Scholar, general)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Step 2              │  Tavily runs each query with
│ Web Search          │  advanced search depth (5 results
│                     │  per query = up to 20 total)
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Step 3              │  GPT-4o-mini reads all snippets
│ Extract Professors  │  and extracts up to 8 structured
│                     │  professor profiles
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Step 4              │  GPT-4o-mini deduplicates and
│ Rank Results        │  ranks professors by relevance
│                     │  to the research interests
└────────┬────────────┘
         ▼
  JSON response → Frontend UI
```

---

## Project Structure

```
professor-finder/
├── main.py          # FastAPI app — routes, schemas, CORS, static serving
├── agent.py         # LangGraph agent — 4-node graph pipeline
├── static/
│   └── index.html   # Frontend UI
├── .env             # API keys (not committed)
├── .env.example     # Template for required keys
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- An **OpenAI API key** (uses `gpt-4o-mini`)
- A **Tavily API key** (free tier at [tavily.com](https://tavily.com))

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd professor-finder
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure API keys**
```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

---

## Running the App

```bash
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

---

## API Reference

### `POST /api/search`

Find professors matching the given university and research interests.

**Request body:**
```json
{
  "university": "MIT",
  "research_interests": "deep learning, computer vision, robotics"
}
```

**Response:**
```json
{
  "university": "MIT",
  "research_interests": "deep learning, computer vision, robotics",
  "search_queries": ["query 1", "query 2", "query 3", "query 4"],
  "count": 5,
  "error": "",
  "professors": [
    {
      "name": "Jane Doe",
      "title": "Associate Professor",
      "department": "EECS",
      "university": "MIT",
      "research_interests": ["deep learning", "object detection"],
      "email": "jdoe@mit.edu",
      "profile_url": "https://...",
      "relevance_summary": "Prof. Doe works on..."
    }
  ]
}
```

### `GET /api/health`

Liveness check.
```json
{ "status": "ok", "agent": "professor_finder_langgraph" }
```

---

## Requirements

```
fastapi
uvicorn
langgraph
langchain-openai
tavily-python
python-dotenv
pydantic
```

Install all at once:
```bash
pip install fastapi uvicorn langgraph langchain-openai tavily-python python-dotenv pydantic
```

---

## Notes & Limitations

- **Hallucination risk:** The agent explicitly instructs GPT-4o-mini not to invent names, but always verify professor details against the linked `profile_url` before reaching out.
- **Result count:** Up to 8 professors are returned per search. Results depend on how much public information Tavily can find.
- **Rate limits:** Tavily free tier allows 1 000 searches/month. Each agent run uses up to 4 searches.
