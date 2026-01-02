# 🤖 BidMate RAG System (입찰메이트 AI)

This project implements a high-performance **RAG (Retrieval-Augmented Generation)** system for analyzing complex Request for Proposal (RFP) documents. It features **Hybrid Search (Semantic + BM25)**, **Context-Aware Chunking**, and an **Async Quantitative Evaluation Pipeline**.

## 🚀 Key Features

- **Hybrid Search Algorithm**: Combines dense vector retrieval (Chroma) with sparse keyword retrieval (BM25) using a custom **Char-Bigram Tokenizer** optimized for Korean technical terms (e.g., "소요예산", "납품").
- **Smart Chunking & Context Injection**: Solves information fragmentation by isolating budget sections and injecting `[Organization] [Project Name]` context into every text chunk.
- **Quantitative Evaluation Framework**: 
    - Auto-generates evaluation datasets (QA pairs) from raw documents using `generate_dataset.py`.
    - Scores performance (1-5 scale) using an **LLM-as-a-Judge** approach via `evaluate.py`.
    - **Current Baseline Score: 3.29 / 5.0** (Top-30 Context Window).
- **Async Pipeline**: Fully asynchronous data processing and evaluation for high throughput.

## 📁 Directory Structure

```
project-2/
├── config/             # Configuration (Chunking size, Retrieval k, etc.)
├── data/               # Raw PDFs, Parsed JSON, and Evaluation Datasets
├── debug_tools/        # Scripts for debugging retrieval, ranking, and DB
├── src/                # Core Source Code
│   ├── pipeline/       # Data Processing (Chunker, Parser)
│   ├── generator.py    # RAG Chain & Prompt Engineering
│   ├── indexer.py      # Vector DB Indexing
│   ├── loader.py       # Data Loading & Context Injection
│   └── retriever.py    # Hybrid Search Implementation (BM25+Chroma)
├── evaluate.py         # Async Evaluation Script (Quantitative Analysis)
├── generate_dataset.py # Evaluation Dataset Generator
├── main.py             # CLI Chat Application
└── README.md           # Project Documentation
```

## 🛠️ Setup & Installation

**Prerequisites**: Python 3.10+, `uv` installed.

1.  **Initialize & Sync**:
    ```bash
    # This project uses uv for dependency management.
    # It will automatically create a virtual environment and install dependencies.
    uv sync
    ```
2.  **Environment Variables**: Create `.env` file.
    ```ini
    OPENAI_API_KEY=sk-...
    UPSTAGE_API_KEY=... # Required for PDF parsing (pipeline step 'parse')
    ```

## 🏃 Usage

### 1. Data Pipeline (Pre-processing)
Run the full data processing pipeline (HWP Convert -> PDF Parse -> Process -> Vector DB):
```bash
python pipeline.py --step all
```
Or run individual steps:
- `convert`: HWP to PDF
- `parse`: PDF to Raw JSON (requires Upstage API)
- `clean`: Cleaning & Chunking (JSON -> JSONL)
- `index`: Build Vector DB (Chroma)

### 2. Chat with RAG (Main Application)
Interact with the system in a conversational mode:
```bash
python main.py
```

### 3. Quantitative Evaluation
Run the automated evaluation against the 100-item technical dataset:
```bash
python evaluate.py --data data/eval_set_100.json --output evaluation_result.csv
```

## 📊 Performance (Benchmark)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Baseline (Top-10)** | 2.90 / 5.0 | Missed details in broad summarization tasks. |
| **Expanded (Top-30)** | **3.29 / 5.0** | Significantly improved coverage for "System Requirements". |

## 🔍 Troubleshooting

- **Retrieval Debugging**: Use scripts in `debug_tools/` to verify chunk ranks:
    ```bash
    python debug_tools/verify_retrieval.py
    ```
- **Configuration**: Adjust `retrieval_k`, `final_k`, and `rerank_weight` in `config/config.yaml`.
