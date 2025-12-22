from google import genai
from pydantic import BaseModel, Field
import os
from typing import List, Dict, Any

class ChunkRelevance(BaseModel):
    is_helpful: bool = Field(description="Whether the chunk contains information relevant to answering the user's query.")
    reasoning: str = Field(description="Brief explanation of why the chunk is helpful or not.")


def extract_unique_chunks(retrieved_chunks: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts and deduplicates chunks from the structured retrieval result.
    """
    unique_chunks_map = {}
    
    def add_chunks_to_map(chunk_list):
        if not chunk_list:
            return
        for chunk in chunk_list:
            c_id = chunk.get("id")
            if c_id and c_id not in unique_chunks_map:
                unique_chunks_map[c_id] = chunk

    # 1. Original Query Chunks
    if retrieved_chunks.get("original_query"):
        add_chunks_to_map(retrieved_chunks["original_query"].get("chunks", []))
        
    # 2. Refined Query Chunks
    if retrieved_chunks.get("refined_query"):
        add_chunks_to_map(retrieved_chunks["refined_query"].get("chunks", []))
        
    # 3. Strategy Chunks
    strategies = retrieved_chunks.get("strategies", {})
    if strategies:
        if strategies.get("hyde_zero_shot_answer"):
            add_chunks_to_map(strategies["hyde_zero_shot_answer"].get("chunks", []))
        if strategies.get("step_back_query"):
            add_chunks_to_map(strategies["step_back_query"].get("chunks", []))
        
        for mq in strategies.get("multi_queries", []):
            add_chunks_to_map(mq.get("chunks", []))
            
        for dq in strategies.get("decomposed_queries", []):
            add_chunks_to_map(dq.get("chunks", []))
    
    unique_chunks = list(unique_chunks_map.values())
    print(f"Unique Chunks to Classify: {len(unique_chunks)}\n")
    return unique_chunks

def classify_chunks(query: str, chunks: List[Dict[str, Any]], model_name: str = "gemini-2.0-flash") -> List[Dict[str, Any]]:
    """
    Classifies each chunk as helpful or not for the given query using an LLM.
    Returns the list of chunks with added 'is_helpful' and 'classification_reasoning' keys.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not set. Skipping classification.")
        return chunks

    client = genai.Client(api_key=api_key)

    classified_chunks = []
    
    print(f"Classifying {len(chunks)} chunks...")

    for chunk in chunks:
        content = chunk.get('content', '')
        
        prompt = f"""
        You are a helpful assistant evaluating the relevance of context chunks for a RAG system.
        Determine if the following chunk contains information useful for answering the query.
        
        Query: {query}
        Chunk: {content}
        """
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ChunkRelevance,
                },
            )
            result = ChunkRelevance.model_validate_json(response.text)
            
            enriched_chunk = chunk.copy()
            enriched_chunk['is_helpful'] = result.is_helpful
            enriched_chunk['classification_reasoning'] = result.reasoning
            classified_chunks.append(enriched_chunk)
            
        except Exception as e:
            print(f"Error classifying chunk {chunk.get('id')}: {e}")
            # Default to False on error to be safe
            enriched_chunk = chunk.copy()
            enriched_chunk['is_helpful'] = False 
            enriched_chunk['classification_reasoning'] = f"Error: {str(e)}"
            classified_chunks.append(enriched_chunk)

    return classified_chunks
