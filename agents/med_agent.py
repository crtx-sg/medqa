# agents/med_agent.py
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult

class MedAgent(BaseAgent):
    def __init__(self, llm: Ollama, tools: Dict, logger):
        super().__init__("MedAgent", llm, tools, logger)

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        function_name = context.get("function")
        if function_name != "answer_medical_question":
            return AgentResult(False, "MedAgent only supports 'answer_medical_question'.")
        
        prompt = f"""
        You are a highly knowledgeable medical expert AI. Address the following user query based on your extensive training on medical literature, guidelines, and clinical knowledge.
        User Query: {user_input}
        Provide a clear, detailed, and helpful response.
        """
        response_text = self.llm.invoke(prompt)
        return AgentResult(True, response_text)

