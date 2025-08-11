# agents/rag_agent.py
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from tools.rag_tool import AdvancedRAGTool

class RAGAgent(BaseAgent):
    def __init__(self, rag_llm: Ollama, rag_tool_instance: Optional[AdvancedRAGTool], logger):
        super().__init__("RAGAgent", rag_llm, {}, logger)
        self.rag_tool = rag_tool_instance
        if self.rag_tool:
            self.rag_tool.llm = rag_llm

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        function_name = context.get("function")
        if function_name != "query_knowledge_base":
            return AgentResult(False, "RAGAgent only supports 'query_knowledge_base'.")
        
        if not self.rag_tool:
            return AgentResult(False, "RAG system is not initialized.")
        
        response_text = self.rag_tool.query(user_input)
        return AgentResult(True, response_text)
