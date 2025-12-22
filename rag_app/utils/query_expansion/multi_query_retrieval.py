from google import genai
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

class MultiQueryResponse(BaseModel):
    queries: List[str] = Field(description="List of 3 generated query versions.")

def generate_multi_queries(query: str, model_name: str = "gemini-2.0-flash") -> List[str]:
    """
    Generates 3 different versions of the user query using Gemini with structured output.
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Generate 3 different versions of the following user query to retrieve relevant documents from a vector database. 
    Use different synonyms and technical perspectives for each.
    
    Original query: "{query}"
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": MultiQueryResponse,
        },
    )

    try:
        result = MultiQueryResponse.model_validate_json(response.text)
        return result.queries
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.text}")
        return []

if __name__ == "__main__":
    load_dotenv()
    test_query = "How does the human body respond to stress?"
    print(f"Generating multiple queries for: {test_query}")
    queries = generate_multi_queries(test_query)
    for q in queries:
        print(f"- {q}")