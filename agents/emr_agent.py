# agents/emr_agent.py
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from tools.clinical_tools import ClinicalDataTools
import json

class EMRAgent(BaseAgent):
    def __init__(self, llm: Ollama, clinical_tools: ClinicalDataTools, logger):
        super().__init__("EMRAgent", llm, {}, logger)
        self.tools = clinical_tools
        self.function_map = {
            "get_patient_image_study": self.tools.get_patient_image_study,
        }

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        function_name = context.get("function")
        params = context.get("params", {})
        target_function = self.function_map.get(function_name)
        if not target_function:
            return AgentResult(False, f"EMRAgent does not have function '{function_name}'.")
        
        result_data = target_function(params)
        
        if "error" in result_data:
            return AgentResult(False, result_data["error"])

        prompt = f"The user asked for an image study. The system retrieved the following data: {json.dumps(result_data)}. Summarize this for the user."
        summary = self.llm.invoke(prompt)
        return AgentResult(True, summary)

