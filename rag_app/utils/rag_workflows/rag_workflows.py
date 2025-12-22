from dotenv import load_dotenv
from utils.query_refinement.query_refinement import refine_query
from utils.query_expansion.query_expansion import expand_query
from utils.similarity_search.request_chunks import request_relevant_chunks
from utils.classify_chunks.classify_chunks import classify_chunks, extract_unique_chunks
from utils.prepare_qa_response.prepare_qa_response import prepare_qa_response



def rag_workflow00(query):

    print(f"Query: {query}\n")

    # QUERY REFINEMENT
    refined_query = refine_query(query)
    print(f"Refined Query: {refined_query}\n")
    
    # QUERY EXPANSION
    expansion_result = expand_query(refined_query)
    candidate_vector_search_queries = {
        "original_query": query,
        "refined_query": refined_query,
        "strategies": expansion_result.get("strategies", {})
    }

    # CHUNK RETRIEVAL
    retrieved_chunks = request_relevant_chunks(candidate_vector_search_queries)
    print(f"Retrieved Chunks: {retrieved_chunks}")
    
    # remove duplicates
    unique_chunks = extract_unique_chunks(retrieved_chunks)

    # CHUNK CLASSIFICATION
    ## helpful or not binary classification
    classified_chunks = classify_chunks(refined_query, unique_chunks)
    helpful_chunks = [c for c in classified_chunks if c.get("is_helpful")]
    print(f"Helpful Chunks: {len(helpful_chunks)}\n")

    # CONDITIONAL RESPONSE GENERATION PATHS
    if not helpful_chunks: # if no helpful chunks
        response = "No helpful chunks found."
        print(response)
        return response
    else: # if helpful chunks
        response = prepare_qa_response(query, helpful_chunks)
        print("--- FINAL RESPONSE ---")
        print(response)
        return response