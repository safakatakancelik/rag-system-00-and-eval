# RAG Application

The brain of the RAG workflows. This directory contains the main application entry point and modular utility components.

## Structure

- **`main.py`** - Main application entry point that uses `rag_workflows` to process queries. It imports and orchestrates the RAG workflow for general use.

- **`utils/`** - Modular utility subdirectories used by `rag_workflows`:
  - **`query_refinement/`** - Query refinement utilities
  - **`query_expansion/`** - Query expansion strategies (HyDE, multi-query retrieval, query decomposition, step-back prompting)
  - **`similarity_search/`** - Vector similarity search and chunk retrieval
  - **`classify_chunks/`** - Chunk classification and filtering utilities
  - **`prepare_qa_response/`** - Response generation utilities
  - **`rag_workflows/`** - RAG workflow orchestration:
    - **`rag_workflows.py`** - General-purpose RAG workflows (uses relative imports)
    - **`rag_workflows_for_evaluation.py`** - Evaluation-optimized RAG workflows (uses absolute imports)

## Important Note

`rag_workflows.py` and `rag_workflows_for_evaluation.py` are essentially the same except one is for general use and one is optimized for the evaluation pipeline. The main difference is in import paths to handle different execution contexts. In the future, if things get more complicated, refactoring is a consideration, but for now it's clear.

## Architecture

The system is designed with decoupled components:
- `main.py` uses `rag_workflows` (imported)
- `rag_workflows` uses the `utils` subdirectories
- Each utility module is independent and reusable

