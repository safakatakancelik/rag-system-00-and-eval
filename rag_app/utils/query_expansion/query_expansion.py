import concurrent.futures
from typing import Dict, Any

# Indirect imports to handle potential path issues if run directly vs as module, 
# though relative imports in a package are usually cleaner. 
# Given the directory structure, relative imports within the package should work 
# if the app is run as a module.
try:
    from .hyde_zero_shot_answer_generator import get_zero_shot_answer
    from .multi_query_retrieval import generate_multi_queries
    from .query_decomposition import decompose_query
    from .step_back_prompting import generate_step_back_query
except ImportError:
    # Fallback/alternative import style if needed, but the above is standard for package
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from hyde_zero_shot_answer_generator import get_zero_shot_answer
    from multi_query_retrieval import generate_multi_queries
    from query_decomposition import decompose_query
    from step_back_prompting import generate_step_back_query

def expand_query(query: str) -> Dict[str, Any]:
    """
    Executes multiple query expansion/understanding strategies in parallel.
    
    Strategies:
    - HyDE (Zero-shot answer)
    - Multi-query generation
    - Query decomposition
    - Step-back prompting
    
    Returns:
        Dict containing the results of each strategy.
    """
    print(f"--- Starting Parallel Strategies for query: '{query}' ---")
    strategies_results = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks
        future_hyde = executor.submit(get_zero_shot_answer, query)
        future_multi = executor.submit(generate_multi_queries, query)
        future_decomp = executor.submit(decompose_query, query)
        future_stepback = executor.submit(generate_step_back_query, query)
        
        # Collect results
        try:
            strategies_results["hyde_zero_shot_answer"] = future_hyde.result()
        except Exception as e:
            print(f"Error in HyDE: {e}")
            strategies_results["hyde_zero_shot_answer"] = None
            
        try:
            strategies_results["multi_queries"] = future_multi.result()
        except Exception as e:
            print(f"Error in Multi-Query: {e}")
            strategies_results["multi_queries"] = []

        try:
            strategies_results["decomposed_queries"] = future_decomp.result()
        except Exception as e:
            print(f"Error in Decomposition: {e}")
            strategies_results["decomposed_queries"] = []

        try:
            strategies_results["step_back_query"] = future_stepback.result()
        except Exception as e:
            print(f"Error in Step-Back: {e}")
            strategies_results["step_back_query"] = None
            
    return {
        "query": query,
        "strategies": strategies_results
    }
