# Document Assistant

LangGraph multi-agent assistant for financial and healthcare documents. Intent
classification routes each turn to a Q&A, summarization, or calculation specialist,
then memory is updated before the turn ends.

Python version: **3.9+** (developed on 3.11).

## What it does

- Classifies user intent (`qa`, `summarization`, `calculation`)
- Searches and reads sample invoices, contracts, and insurance claims
- Answers questions with source document IDs
- Summarizes documents
- Runs calculations through a validated calculator tool
- Persists short-term state with LangGraph `InMemorySaver` keyed by `thread_id`
- Saves session metadata under `sessions/`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add OPENAI_API_KEY or VOCAREUM_API_KEY
```

If you are on the Udacity / Vocareum proxy, set:

```
VOCAREUM_API_KEY=...
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

## Run

```bash
python main.py
```

Example prompts:

- `What's the total amount in invoice INV-001?`
- `Summarize the service agreement`
- `Calculate the sum of all invoice totals`
- `Find documents with amounts over $50,000`

Type `/docs` to list documents, `/quit` to exit.

## Tests

```bash
pytest -q
```

Offline tests cover schemas, the calculator, retrieval, prompt routing, intent
mapping, and graph structure. They do not call an LLM.

## Packages

See `requirements.txt`. Notable versions aligned with the course examples:

- langgraph>=0.5.4
- langchain>=0.3.27
- langchain-openai>=0.3.28
- pytest>=8.3.0

## Design

Graph, state, and routing notes: `docs/ARCHITECTURE.md`.
