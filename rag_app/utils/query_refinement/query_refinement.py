from google import genai
from pydantic import BaseModel, Field
import os

# Class to enforce structured output
class RefinedQueryResponse(BaseModel):
    refined_query: str = Field(description="The refined, standalone search query.")



def refine_query(user_query: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Refines a user query by removing conversational noise and extracting the core intent, without any chat history.
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are a search query refiner within a RAG system.
    Process the user query and turn it into a clear, concise, and standalone search query.
    Remove any conversational filler, polite introductory phrases, or unrelated noise.
    Capture the core intent of the user.

    User Input: {user_query}
    """

    # request a response
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RefinedQueryResponse, # enforce structured output
        },
    )

    # Validate response format
    try:
        result = RefinedQueryResponse.model_validate_json(response.text)
        return result.refined_query
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.text}")
        return ""

if __name__ == "__main__":
    query = "Hey there, I was wondering if you could tell me what is the capital of France?"
    refined_query = refine_query(query)
    print(f"Refined Query: {refined_query}")
