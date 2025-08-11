# agents/web_agent.py
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger
from routing_config import WEB_AGENT_SOURCES, WEB_AGENT_SCRAPE_SOURCES
import requests
from bs4 import BeautifulSoup

class WebAgent(BaseAgent):
    def __init__(self, llm: Ollama, logger: AppLogger):
        super().__init__("WebAgent", llm, {}, logger)
        self.function_map = {
            "scrape_predefined_links": self.scrape_predefined_links,
            "search_predefined_links": self.search_predefined_links,
        }

    def _scrape_links(self, links: list[str]) -> str:
        # ... (scraping logic remains the same)
        consolidated_text = ""
        for url in links:
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                consolidated_text += f"\n--- Content from {url} ---\n{soup.get_text(strip=True, separator=' ')[:1500]}"
            except Exception as e:
                consolidated_text += f"\n--- Failed to scrape {url}: {e} ---"
        return consolidated_text

    def scrape_predefined_links(self, params: Dict) -> str:
        content = self._scrape_links(list(WEB_AGENT_SCRAPE_SOURCES.values()))
        prompt = f"Summarize the key points from the following content scraped from predefined news websites:\n\n{content}"
        return self.llm.invoke(prompt)

    def search_predefined_links(self, user_input: str) -> str:
        content = self._scrape_links(list(WEB_AGENT_SOURCES.values()))
        prompt = f"Based on content from predefined sources, answer the query: '{user_input}'.\n\nContent:\n{content}"
        return self.llm.invoke(prompt)

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        function_name = context.get("function")
        target_function = self.function_map.get(function_name)
        if not target_function:
            return AgentResult(False, f"WebAgent does not have function '{function_name}'.")
        
        # Pass the original user_input to the search function
        if function_name == "search_predefined_links":
            response_text = target_function(user_input)
        else:
            response_text = target_function(context.get("params", {}))
            
        return AgentResult(True, response_text)
