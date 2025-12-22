from google import genai
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

class DecomposedQueryResponse(BaseModel):
    decomposed_queries: List[str] = Field(description="List of 3 distinct, specific search strings.")

def decompose_query(query: str, model_name: str = "gemini-2.0-flash") -> List[str]:
    """
    Decomposes a user query into three distinct, specific search strings using Gemini with structured output.
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are a RAG Query Optimizer. Your goal is to rewrite a user query into three 
    distinct, highly specific search strings that will be used for a vector database 
    similarity search.

    Rules:
    1. One query should focus on core technical definitions.
    2. One query should focus on the 'How-to' or process.
    3. One query should use academic/industry synonyms to bridge the semantic gap.

    User Query: "{query}"
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": DecomposedQueryResponse,
        },
    )

    try:
        result = DecomposedQueryResponse.model_validate_json(response.text)
        return result.decomposed_queries
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.text}")
        return []

if __name__ == "__main__":
    load_dotenv()
    test_query = "Optimizing SQL queries for large datasets"
    print(f"Decomposing query: {test_query}")
    queries = decompose_query(test_query)
    for q in queries:
        print(f"- {q}")