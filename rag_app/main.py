
from utils.rag_workflows.rag_workflows import rag_workflow00
from dotenv import load_dotenv

def main():
    # load environment variables    
    load_dotenv()
    
    # user query to be processed
    query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"

    try:
        # call defined rag workflow with the user query
        response = rag_workflow00(query)
        return response
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
