
import json
import os
import sys
from typing import List, Dict, Any

# Ensure project root is in path to import vector_db
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from vector_db.src.client import VectorDBClient

def _fetch_chunks_for_query(client: VectorDBClient, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Helper to fetch chunks for a single query."""
    if not query_text:
        return []
        
    try:
        results = client.similarity_search(query_text=query_text, top_k=top_k)
        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        dists = results['distances'][0]
        
        chunks = []
        for i, chunk_id in enumerate(ids):
            chunks.append({
                "id": chunk_id,
                "content": docs[i],
                "source": metas[i].get("source", "Unknown") if metas and metas[i] else "Unknown",
                "distance": dists[i]
            })
        return chunks
    except Exception as e:
        print(f"Error searching for query '{query_text}': {e}")
        return []

def request_relevant_chunks(optimization_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the structured output from the RAG query optimizer, performs similarity searches,
    and returns a structured dictionary organizing retrieved chunks by method.
    """
    
    # Initialize Vector DB Client
    persist_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vector_db', 'chroma_db')
    client = VectorDBClient(persist_directory=persist_dir)
    
    # Structure to hold the final result
    structured_result = {
        "original_query": None,
        "refined_query": None,
        "strategies": {
            "hyde_zero_shot_answer": None,
            "step_back_query": None,
            "multi_queries": [],
            "decomposed_queries": []
        }
    }

    print("--- Starting Chunk Retrieval (Structured) ---")

    # 1. Original Query
    orig_q = optimization_result.get("original_query")
    if orig_q:
        print("Fetching for Original Query...")
        structured_result["original_query"] = {
            "query_text": orig_q,
            "chunks": _fetch_chunks_for_query(client, orig_q)
        }

    # 2. Refined Query
    ref_q = optimization_result.get("refined_query")
    if ref_q:
        print("Fetching for Refined Query...")
        structured_result["refined_query"] = {
            "query_text": ref_q,
            "chunks": _fetch_chunks_for_query(client, ref_q)
        }

    # 3. Strategies
    strategies = optimization_result.get("strategies", {})

    # HyDE
    hyde_q = strategies.get("hyde_zero_shot_answer")
    if hyde_q:
        print("Fetching for HyDE...")
        structured_result["strategies"]["hyde_zero_shot_answer"] = {
            "query_text": hyde_q,
            "chunks": _fetch_chunks_for_query(client, hyde_q)
        }
        
    # Step Back
    step_back_q = strategies.get("step_back_query")
    if step_back_q:
        print("Fetching for Step-Back...")
        structured_result["strategies"]["step_back_query"] = {
            "query_text": step_back_q,
            "chunks": _fetch_chunks_for_query(client, step_back_q)
        }

    # Multi Queries
    multi_queries = strategies.get("multi_queries", [])
    if multi_queries:
        print(f"Fetching for {len(multi_queries)} Multi-Queries...")
        for q in multi_queries:
            structured_result["strategies"]["multi_queries"].append({
                "query_text": q,
                "chunks": _fetch_chunks_for_query(client, q)
            })

    # Decomposed Queries
    decomp_queries = strategies.get("decomposed_queries", [])
    if decomp_queries:
        print(f"Fetching for {len(decomp_queries)} Decomposed Queries...")
        for q in decomp_queries:
            structured_result["strategies"]["decomposed_queries"].append({
                "query_text": q,
                "chunks": _fetch_chunks_for_query(client, q)
            })

    # Save to JSON
    output_file = "retrieved_chunks.json"
    try:
        with open(output_file, "w") as f:
            json.dump(structured_result, f, indent=4)
        print(f"--- Chunk retrieval complete. Results saved to {output_file} ---")
    except Exception as e:
        print(f"Error saving retrieved chunks: {e}")
        
    return structured_result

if __name__ == "__main__":
    # Test with the existing result file if it exists
    input_file = "query_optimization_result.json"
    if os.path.exists(input_file):
        with open(input_file, "r") as f:
            opt_result = json.load(f)
        
        chunks = request_relevant_chunks(opt_result)
    else:
        print(f"File {input_file} not found. Run rag_query_optimizer.py first.")
