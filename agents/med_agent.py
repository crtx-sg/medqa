from typing import Dict, Any, Optional, Callable
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger

# --- Med Agent Tools (can be expanded later) ---

def get_med_tools(logger: AppLogger) -> Dict[str, Callable]:
    """
    Factory for MedAgent tools. Currently empty as the agent's primary role
    is direct Q&A with its specialized LLM. Can be expanded to include
    tools for PubMed API calls, etc.
    """
    logger.log("MedAgent tools initialized (currently none).")
    return {}

# --- Med Agent Definition ---

class MedAgent(BaseAgent):
    """
    Handles general medical domain queries using a specialized LLM. It answers
    questions related to clinical research, medical guidelines, and USMLE-style problems.
    """
    def __init__(self, llm: Ollama, tools: Dict[str, Callable], logger: AppLogger):
        super().__init__("MedAgent", llm, tools, logger)

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        self.logger.log(f"{self.name} is handling: '{user_input}'")
        
        # This agent's primary purpose is to leverage its specialized model.
        # It wraps the user query in a prompt designed for a medical expert LLM.
        prompt = f"""
        You are a highly knowledgeable medical expert AI. Your purpose is to provide accurate, evidence-based answers to medical questions.
        Address the following user query based on your extensive training on medical literature, guidelines, and clinical knowledge.

        User Query: {user_input}

        Provide a clear, detailed, and helpful response.
        """
        
        self.logger.log("Invoking specialized medical LLM...")
        llm_resp = self.llm.invoke(prompt)
        self.logger.log("Received response from medical LLM.")
        
        # This agent typically does not produce plots, so plot_data is None.
        return AgentResult(True, llm_resp)

