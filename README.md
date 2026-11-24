# FinSight — AI-Powered Financial Research Assistant

FinSight is an AI-powered financial research assistant designed to answer questions about Indian companies using their annual reports. It combines Retrieval-Augmented Generation (RAG), hybrid information retrieval, cross-encoder reranking, and structured financial data access to produce evidence-grounded responses with source citations.

The project currently includes an end-to-end working vertical slice for six major Indian IT companies.

## Key Features

- **Annual report ingestion:** Extracts text from PDF annual reports, splits it into overlapping chunks, and prepares the content for retrieval.
- **Hybrid retrieval:** Combines dense vector retrieval and BM25 keyword search.
- **Cross-encoder reranking:** Reorders retrieved evidence to improve relevance.
- **Company-aware retrieval:** Restricts searches to the selected company's documents.
- **Evidence-grounded generation:** Generates answers using retrieved evidence and validates supporting quotations.
- **Source citations:** Returns document names, page numbers, chunk identifiers, and supporting evidence.
- **Financial data access:** Includes a SQL-backed data layer for structured financial information.
- **REST API:** Exposes the question-answering pipeline through FastAPI.
- **Interactive interface:** Provides a Streamlit chat interface.
- **Evaluation framework:** Includes retrieval evaluation and answer-evaluation components.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11 |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| PDF processing | PyMuPDF |
| Dense retrieval | Sentence Transformers |
| Keyword retrieval | BM25 |
| Retrieval fusion | Reciprocal Rank Fusion |
| Reranking | Cross-encoder |
| LLM runtime | Ollama |
| Data validation | Pydantic |
| Database | SQL |
| Testing | pytest |
| Code quality | Ruff |
| Dependency management | uv |

## Architecture

```text
                  User
                   |
                   v
          Streamlit Application
                   |
                   v
              FastAPI
                   |
                   v
             Agent Pipeline
                   |
          +--------+--------+
          |                 |
          v                 v
    Company Scoping    SQL Data Layer
          |
          v
     Hybrid Retrieval
          |
     +----+----+
     |         |
     v         v
  Dense      BM25
 Search     Search
     |         |
     +----+----+
          |
          v
    Result Fusion
          |
          v
  Cross-Encoder Reranking
          |
          v
   Evidence Selection
          |
          v
   Answer Generation
          |
          v
  Evidence Validation
          |
          v
  Answer + Citations
```

### Document ingestion pipeline

```text
Annual Report PDFs
       |
       v
   PDF Parsing
       |
       v
 Text Extraction
       |
       v
  Text Chunking
       |
       v
 Embedding Generation
       |
       v
 Retrieval Index
```

## Current Dataset

The current annual-report retrieval vertical slice covers six Indian IT companies:

| Company | NSE Symbol |
|---|---|
| HCL Technologies | HCLTECH |
| Infosys | INFY |
| LTIMindtree / LTM | LTM |
| Tata Consultancy Services | TCS |
| Tech Mahindra | TECHM |
| Wipro | WIPRO |

The company reference master contains 100 Nifty 100 constituents. The current indexed annual-report dataset covers the six IT companies listed above; it does not yet represent complete Nifty 100 annual-report coverage.

The current retrieval index contains:

- **5,019 document chunks**
- **384-dimensional embeddings**
- **Six company-specific document collections**
- **40 benchmark questions**

## Retrieval Evaluation

FinSight was evaluated using a 40-question benchmark with company-scoped retrieval.

### Hybrid retrieval and reranking baseline

| Metric | Result |
|---|---:|
| Page Recall@5 | 76.25% |
| Page Recall@10 | 87.50% |
| Page Recall@50 | 100.00% |
| Chunk Recall@5 | 70.83% |
| Chunk Recall@10 | 83.33% |
| Chunk Recall@50 | 98.75% |
| Reranked Chunk Recall@10 | 88.75% |
| Reranked MRR@10 | 0.6590 |
| Reranked NDCG@10 | 0.7105 |

The reranking stage improved chunk Recall@10 from 83.33% to 88.75% on this benchmark.

**Evaluation scope:** These results measure retrieval relevance, not answer correctness, factual accuracy, or faithfulness. They should not be interpreted as an answer-accuracy score.

## Project Structure

```text
FinSight/
|
├── app/
│   └── streamlit_app.py
|
├── data/
│   ├── reference/
│   ├── processed/
│   └── eval/
|
├── docs/
|
├── scripts/
|
├── src/
│   └── finsight/
│       ├── agent/
│       │   ├── factory.py
│       │   └── pipeline.py
│       │
│       ├── api/
│       │   └── main.py
│       │
│       ├── evaluation/
│       │
│       ├── generation/
│       │
│       ├── ingestion/
│       │
│       ├── llm/
│       │
│       ├── retrieval/
│       │
│       └── sql/
│
├── tests/
├── pyproject.toml
└── uv.lock
```

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/woohasri-ai/FinSight.git
cd FinSight
```

### 2. Create and activate the virtual environment

```bash
uv sync --group dev
```

For PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Start Ollama

Install Ollama separately and download the model used by the current local setup:

```bash
ollama pull qwen2.5:1.5b
```

Make sure the Ollama service is running before sending questions to the application.

### 4. Start the FastAPI backend

```bash
uv run uvicorn finsight.api.main:app --host 127.0.0.1 --port 8000
```

API health check:

```text
http://127.0.0.1:8000/health
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit frontend

Open another terminal in the project directory:

```bash
uv run streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

**Data requirement:** The application also depends on its prepared document indexes and structured database. A fresh clone may require the project data-preparation pipeline to be run before questions can be answered. Do not assume that installing Python dependencies alone creates these artifacts.

## Running Tests

Run the complete test suite:

```bash
uv run pytest -q
```

The current verified test result is:

**304 tests passed, 0 failures.**

Run Ruff:

```bash
uv run ruff check .
```

## Example Question

```text
What was TCS's consolidated net profit in FY2026?
```

A verified test against the current TCS annual-report index returned the financial statement line:

**Profit after tax attributable to shareholders of the Company: ₹52,820 crore (FY2026).**

The response included its source document, page number, chunk identifier, and supporting evidence.

## Reliability and Limitations

FinSight is designed to prioritize evidence over unsupported answers.

- Responses are grounded in retrieved annual-report evidence.
- Supporting quotations are validated against source evidence.
- Insufficient evidence can result in an abstention rather than an invented answer.
- Financial coverage depends on the reports and structured records currently available.
- Retrieval benchmark scores do not establish answer correctness.
- The current indexed annual-report coverage is limited to six Indian IT companies.

## Future Scope

- Expand annual-report coverage across additional Nifty 100 companies.
- Improve structured financial data extraction and coverage.
- Expand the benchmark and evaluate answer correctness and faithfulness.
- Improve financial table interpretation and answer presentation.
- Extend the application with additional financial analysis capabilities.

## Author

**Wooha Sri**

B.Tech — Artificial Intelligence and Machine Learning

[LinkedIn](https://www.linkedin.com/in/woohasri-midde-a289b4297/)

[GitHub](https://github.com/woohasri-ai)

---

*FinSight — Turning financial reports into evidence-backed answers.*
