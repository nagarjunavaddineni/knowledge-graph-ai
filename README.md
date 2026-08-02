# Knowledge Graph AI

A Graph RAG (Retrieval-Augmented Generation) assistant that answers natural-language questions about people, projects, and customers by querying a Neo4j knowledge graph and grounding an LLM's response in the retrieved graph context.

## How it works

1. Company data (people, projects, customers, and their relationships) is loaded into a Neo4j graph database.
2. A user asks a question about a person and a customer (e.g. "How is John connected to Customer X?").
3. The app runs a Cypher query to find the relationship path between them (`Person -[:MANAGES]-> Project -[:SERVES]-> Customer`).
4. The retrieved graph facts are passed as context to an OpenAI model, which answers the question using only that context.

## Project structure

```
.
├── main.py                # CLI entry point for the Graph RAG assistant
├── graph_rag.py            # Builds prompt/context from graph results and calls OpenAI
├── graph_db.py              # Neo4j driver setup and query execution helpers
├── build_graph.py           # Loads data/company_data.json and populates the Neo4j graph
├── data/
│   └── company_data.json    # Sample people/projects/customers/relationships dataset
├── app/                     # In-progress production module layout
│   ├── config.py             # Environment-based settings (Settings.from_environment)
│   ├── logging_config.py     # Centralized logging configuration
│   ├── api/                  # (planned) API layer
│   ├── database/             # (planned) Database access layer
│   ├── ingestion/             # (planned) Data ingestion pipeline
│   ├── retrieval/             # (planned) Graph retrieval logic
│   ├── services/              # (planned) Business/service layer
│   └── ui/                    # (planned) UI layer
├── legacy/legacy/            # Earlier Streamlit-based prototype (app.py, graph_search.py)
├── docs/                     # Documentation (empty)
├── scripts/                  # Utility scripts (empty)
└── tests/                    # Tests (empty)
```

> Note: `app/` currently holds the scaffolding for a production-oriented refactor of the top-level scripts (`graph_rag.py`, `graph_db.py`, `build_graph.py`). The two currently coexist.

## Prerequisites

- Python 3.11+
- A Neo4j database (e.g. [Neo4j AuraDB](https://neo4j.com/cloud/platform/aura-graph-database/) free tier)
- An OpenAI API key

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your credentials:

   ```bash
   cp .env.example .env
   ```

   | Variable        | Description                                  | Default          |
   |-----------------|-----------------------------------------------|------------------|
   | `NEO4J_URI`      | Connection URI for your Neo4j instance        | —                |
   | `NEO4J_USERNAME` | Neo4j username                                | `neo4j`          |
   | `NEO4J_PASSWORD` | Neo4j password                                | —                |
   | `OPENAI_API_KEY` | OpenAI API key                                | —                |
   | `OPENAI_MODEL`   | OpenAI model used for answering questions     | `gpt-4.1-mini`   |
   | `APP_ENV`        | Application environment (`development`, etc.) | `development`    |
   | `LOG_LEVEL`      | Logging verbosity                             | `INFO`           |

## Usage

**1. Build the graph** — load the sample dataset into Neo4j:

```bash
python build_graph.py
```

**2. Ask questions** via the CLI:

```bash
python main.py
```

You'll be prompted for a person's name, a customer's name, and your question. The assistant looks up the relationship path in the graph and answers using only that context.

**3. (Optional) Streamlit UI** — a Streamlit-based prototype is available under `legacy/legacy/app.py`:

```bash
streamlit run legacy/legacy/app.py
```

## Sample data

`data/company_data.json` contains a small sample dataset (people, projects, customers, and their relationships) you can use to try out the assistant out of the box, or replace with your own.
