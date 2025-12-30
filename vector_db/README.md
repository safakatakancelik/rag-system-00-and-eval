# Vector Database

ChromaDB vector database client with helper functions for chunking, embedding, and storage operations.

## Source

This directory is a clone of the [Vector-DB-Explorer](https://github.com/safakatakancelik/Vector-DB-Explorer/tree/main) repository.

## Structure

- **`src/`** - Core vector database functionality:
  - `client.py` - ChromaDB client interface
  - `chunking.py` - Text chunking utilities
  - `embeddings.py` - Embedding generation
  - `loaders.py` - Data loading utilities
  - `storage.py` - Storage management

- **`chroma_db/`** - ChromaDB data storage directory (contains vector database files)

- **`tests/`** - Test files for loaders and search functionality

