from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv



def get_zero_shot_answer(query: str, model_name: str = "gemini-2.0-flash"):
    """
    Generates a zero-shot answer to a user query.
    serves to the purpose of creating a fake answer.
    HyDE (Hypothetical Document Embeddings)
    """
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an helpful assistant that can answer user queries.
    Give a concise answer that captures the essence of the user query.
    Do not include any additional information or introductory phrases.
    Summarize the answer in a single sentence.
    Focus on using specific terminology and facts that would likely appear in a textbook or technical documentation.

    Now, process this User Query:
    "{query}"
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "maxOutputTokens": 600
        },
    )

    return response.text

if __name__ == "__main__":
    load_dotenv()
    test_query = "How does the human body respond to stress?"
    print(f"Responding to the query: {test_query}")
    response = get_zero_shot_answer(test_query)
    print(type(response), response)