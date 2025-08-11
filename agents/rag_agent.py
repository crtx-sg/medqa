from typing import Dict, Any, Optional, Callable
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger
from utils.rag_tool import AdvancedRAGTool

# --- RAG Agent Definition ---

class RAGAgent(BaseAgent):
    """
    Handles queries by leveraging an advanced, configurable RAG tool.
    This agent is initialized with a pre-configured RAG system.
    """
    def __init__(self, llm: Ollama, rag_tool_instance: Optional[AdvancedRAGTool], logger: AppLogger):
        # The 'tools' dict is now less relevant as the main tool is the RAG instance itself.
        # We pass the instance directly.
        super().__init__("RAGAgent", llm, {}, logger)
        self.rag_tool = rag_tool_instance

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        self.logger.log(f"{self.name} is handling: '{user_input}'")
        
        if not self.rag_tool:
            return AgentResult(False, "The RAG system has not been initialized. Please configure it in the sidebar and load documents.")

        # The core logic is now delegated entirely to the RAG tool's query method.
        response_text = self.rag_tool.query(user_input)
        
        return AgentResult(True, response_text)

