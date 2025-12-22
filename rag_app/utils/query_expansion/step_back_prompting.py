from google import genai
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

class StepBackQueryResponse(BaseModel):
    step_back_query: str = Field(description="The broader, high-level conceptual question.")

def generate_step_back_query(query: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Generates a broader, high-level conceptual question using Gemini with structured output.
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    Given the following specific user query, generate a broader, high-level conceptual question that would provide the necessary background information to answer it.

    Specific Query: "{query}"
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": StepBackQueryResponse,
        },
    )

    try:
        result = StepBackQueryResponse.model_validate_json(response.text)
        return result.step_back_query
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Raw response: {response.text}")
        return ""

if __name__ == "__main__":
    load_dotenv()
    test_query = "Why is my docker container exiting immediately?"
    print(f"Generating step back query for: {test_query}")
    step_back = generate_step_back_query(test_query)
    print(f"Step Back Query: {step_back}")