"""Actor component that executes tools/actions."""
import requests
import pandas as pd
from typing import Dict, Optional, List
from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS
import os

class Actor:
    """Executes tools: web search, API calls, CSV lookups."""
    
    def __init__(self):
        self.tavily_api_key = TAVILY_API_KEY
    
    def web_search(self, query: str) -> Dict:
        """Perform web search using Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            Dict with search results
        """
        if not self.tavily_api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured",
                "result": ""
            }
        
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": MAX_SEARCH_RESULTS
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Format results
            results_text = ""
            if 'results' in data:
                for i, result in enumerate(data['results'], 1):
                    results_text += f"\n[{i}] {result.get('title', 'No title')}\n"
                    results_text += f"URL: {result.get('url', 'N/A')}\n"
                    results_text += f"Content: {result.get('content', '')[:500]}...\n"
            
            return {
                "success": True,
                "tool": "web_search",
                "result": results_text,
                "raw_data": data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": f"Error performing web search: {str(e)}"
            }
    
    def api_call(self, endpoint: str, method: str = "GET", 
                 params: Optional[Dict] = None, 
                 data: Optional[Dict] = None) -> Dict:
        """Make a REST API call.
        
        Args:
            endpoint: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body data
            
        Returns:
            Dict with API response
        """
        try:
            if method.upper() == "GET":
                response = requests.get(endpoint, params=params, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(endpoint, json=data, params=params, timeout=10)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported method: {method}",
                    "result": ""
                }
            
            response.raise_for_status()
            
            try:
                result_data = response.json()
            except:
                result_data = response.text
            
            return {
                "success": True,
                "tool": "api_call",
                "result": str(result_data),
                "raw_data": result_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": f"Error calling API: {str(e)}"
            }
    
    def csv_lookup(self, csv_path: str, query: str, 
                   search_columns: Optional[List[str]] = None) -> Dict:
        """Look up data from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            query: Search query (searches in all columns or specified columns)
            search_columns: List of column names to search in (None = all columns)
            
        Returns:
            Dict with matching rows
        """
        try:
            if not os.path.exists(csv_path):
                return {
                    "success": False,
                    "error": f"CSV file not found: {csv_path}",
                    "result": ""
                }
            
            df = pd.read_csv(csv_path)
            
            # Search in specified columns or all columns
            if search_columns:
                search_cols = [col for col in search_columns if col in df.columns]
            else:
                search_cols = df.columns.tolist()
            
            # Filter rows containing query (case-insensitive)
            mask = df[search_cols].astype(str).apply(
                lambda x: x.str.contains(query, case=False, na=False)
            ).any(axis=1)
            
            matching_rows = df[mask]
            
            if len(matching_rows) == 0:
                return {
                    "success": True,
                    "tool": "csv_lookup",
                    "result": f"No matching rows found for query: {query}",
                    "raw_data": None
                }
            
            # Format results
            result_text = f"Found {len(matching_rows)} matching row(s):\n\n"
            result_text += matching_rows.to_string(index=False)
            
            return {
                "success": True,
                "tool": "csv_lookup",
                "result": result_text,
                "raw_data": matching_rows.to_dict('records')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "result": f"Error looking up CSV: {str(e)}"
            }
    
    def execute_tool(self, tool_name: str, tool_query: str, **kwargs) -> Dict:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            tool_query: Query/input for the tool
            **kwargs: Additional tool-specific parameters
            
        Returns:
            Tool execution result
        """
        if tool_name == "web_search":
            return self.web_search(tool_query)
        elif tool_name == "api_call":
            endpoint = kwargs.get("endpoint", tool_query)
            method = kwargs.get("method", "GET")
            params = kwargs.get("params")
            data = kwargs.get("data")
            return self.api_call(endpoint, method, params, data)
        elif tool_name == "csv_lookup":
            # Default to data/prices.csv, but allow override
            csv_path = kwargs.get("csv_path", os.path.join("data", "prices.csv"))
            # If relative path doesn't exist, try absolute from project root
            if not os.path.exists(csv_path):
                csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prices.csv")
            search_columns = kwargs.get("search_columns")
            return self.csv_lookup(csv_path, tool_query, search_columns)
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "result": ""
            }
