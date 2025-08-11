from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger
from routing_config import WEB_AGENT_SOURCES, WEB_AGENT_SCRAPE_SOURCES
import requests
from bs4 import BeautifulSoup

class WebAgent(BaseAgent):
    """
    Acts as a dispatcher for web-based tools based on commands from the orchestrator.
    """
    def __init__(self, llm: Ollama, logger: AppLogger):
        super().__init__("WebAgent", llm, {}, logger)
        # Map function names to the actual methods
        self.function_map = {
            "scrape_predefined_links": self.scrape_predefined_links,
            "search_predefined_links": self.search_predefined_links,
        }

    def _scrape_links(self, links: list[str]) -> str:
        """Internal method to scrape a list of provided URLs."""
        consolidated_text = ""
        for url in links:
            self.logger.log(f"WEBTOOL: Scraping {url}")
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                # Remove common non-content tags
                for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                    script.extract()
                consolidated_text += f"\n--- Content from {url} ---\n{soup.get_text(strip=True, separator=' ')[:2000]}"
            except Exception as e:
                consolidated_text += f"\n--- Failed to scrape {url}: {e} ---"
        return consolidated_text

    def scrape_predefined_links(self, params: Dict) -> str:
        """Tool function to scrape a predefined list of news/update websites."""
        content = self._scrape_links(list(WEB_AGENT_SCRAPE_SOURCES.values()))
        prompt = f"Summarize the key points from the following content scraped from predefined news websites:\n\n{content}"
        return self.llm.invoke(prompt)

    def search_predefined_links(self, user_input: str) -> str:
        """Tool function to search a predefined list of reference websites."""
        content = self._scrape_links(list(WEB_AGENT_SOURCES.values()))
        prompt = f"Based on content from predefined reference sources, answer the query: '{user_input}'.\n\nContent:\n{content}"
        return self.llm.invoke(prompt)

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        """
        Executes the function specified by the orchestrator.
        """
        function_name = context.get("function")
        params = context.get("params", {})
        
        self.logger.log(f"WebAgent executing function: {function_name} with params: {params}")

        target_function = self.function_map.get(function_name)
        if not target_function:
            return AgentResult(False, f"WebAgent does not have a function named '{function_name}'.")
        
        try:
            # The 'search' function needs the original user query, while 'scrape' does not.
            if function_name == "search_predefined_links":
                response_text = target_function(user_input)
            else:
                response_text = target_function(params)
            
            return AgentResult(True, response_text)
        except Exception as e:
            self.logger.log(f"Error executing {function_name}: {e}")
            return AgentResult(False, f"An error occurred while executing the web task: {e}")

