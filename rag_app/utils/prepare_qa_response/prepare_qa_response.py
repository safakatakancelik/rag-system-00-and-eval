from google import genai
from pydantic import BaseModel, Field
import os
from typing import List, Dict, Any

class QAResponse(BaseModel):
    answer: str = Field(description="The comprehensive answer to the user's query based on the context.")

def prepare_qa_response(query: str, helpful_chunks: List[Dict[str, Any]], model_name: str = "gemini-2.0-flash") -> str:
    """
    Generates a response to the user's query using the helpful chunks as context.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set."

    client = genai.Client(api_key=api_key)

    context_text = ""
    for i, chunk in enumerate(helpful_chunks):
        source = chunk.get('source', 'Unknown')
        content = chunk.get('content', '').strip()
        context_text += f"\n[Chunk {i+1} | Source: {source}]\n{content}\n"

    prompt = f"""
    You are a knowledgeable assistant. Answer the user's query using ONLY the provided context chunks.
    If the answer cannot be found in the context, say so.
    
    Query: {query}
    
    Context:
    {context_text}
    """

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QAResponse,
            },
        )
        result = QAResponse.model_validate_json(response.text)
        return result.answer
    except Exception as e:
        print(f"Error generating QA response: {e}")
        return "I encountered an error generating the response."
