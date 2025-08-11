import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, Callable
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger
from routing_config import WEB_AGENT_SOURCES # Correctly import the renamed variable

# --- Web Agent Tools ---

def tool_web_scrape(params: Dict[str, Any], logger: AppLogger) -> Dict[str, Any]:
    """Scrapes a given URL and returns the text content."""
    url = params.get("url")
    query = params.get("query")
    if not url:
        return {"error": "No URL provided."}
        
    logger.log(f"TOOL: tool_web_scrape called for URL: {url}")
    try:
        if "pubmed" in url:
            search_url = f"{url}?term={query.replace(' ', '+')}"
        else:
            search_url = url

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        text = soup.get_text(separator='\n', strip=True)
        
        logger.log(f"Successfully scraped {len(text)} characters from {url}.")
        return {"url": url, "content": text[:4000]}
    except requests.RequestException as e:
        logger.log(f"ERROR: Web scraping failed for {url}: {e}")
        return {"error": str(e)}

def get_web_tools(logger: AppLogger) -> Dict[str, Callable]:
    """Factory to create the web scraping tool."""
    return {
        "web_scrape": lambda p: tool_web_scrape(p, logger),
    }

# --- Web Agent Definition ---

class WebAgent(BaseAgent):
    """
    Handles queries requiring real-time information by scraping the web.
    """
    def __init__(self, llm: Ollama, tools: Dict[str, Callable], logger: AppLogger):
        super().__init__("WebAgent", llm, tools, logger)

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        self.logger.log(f"{self.name} is handling: '{user_input}'")
        
        retrieved_context = ""
        metadata = {"sources": []}
        
        # Log all sources that will be searched
        self.logger.log(f"WebAgent will search the following sources: {list(WEB_AGENT_SOURCES.keys())}")
        
        # Scrape ALL configured web sources and consolidate the results
        for source_key, url in WEB_AGENT_SOURCES.items():
            self.logger.log(f"Searching source: {source_key} at URL: {url}")
            scrape_result = self.tools["web_scrape"]({"url": url, "query": user_input})
            
            if "content" in scrape_result and not "error" in scrape_result:
                retrieved_context += f"--- Information from {source_key.replace('_', ' ').title()} ---\n{scrape_result['content']}\n\n"
                metadata["sources"].append(url)
            else:
                self.logger.log(f"Could not retrieve content from {source_key}. Error: {scrape_result.get('error')}")

        if not retrieved_context:
            return AgentResult(False, "Failed to scrape any information from the configured web sources.")

        prompt = f"""
        You are a helpful research assistant. Based on the following consolidated information retrieved from the web, answer the user's query.
        
        Retrieved Context:
        {retrieved_context}
        
        User Query: {user_input}
        
        Answer:
        """
        llm_resp = self.llm.invoke(prompt)
        return AgentResult(True, llm_resp, metadata=metadata)

