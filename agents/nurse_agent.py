# agents/nurse_agent.py
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from tools.clinical_tools import ClinicalDataTools
from tools.ews_analyzer import EWSTrendAnalyzer
import datetime
import json
import random

class NurseAgent(BaseAgent):
    """
    Acts as a dispatcher for clinical tools, intelligently formatting responses
    based on the user's intent.
    """
    def __init__(self, llm: Ollama, clinical_tools: ClinicalDataTools, logger):
        super().__init__("NurseAgent", llm, {}, logger)
        self.tools = clinical_tools
        self.function_map = {
            "get_patient_info": self.tools.get_patient_info,
            "get_all_patients_info": self.tools.get_all_patients_info,
            "get_patient_alarms": self.tools.get_patient_alarms,
            "get_critical_patients": self.tools.get_critical_patients,
            "get_patient_vitals_trend": self.tools.get_patient_vitals_trend,
        }

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        function_name = context.get("function")
        params = context.get("params", {})
        
        self.logger.log(f"NurseAgent executing function: {function_name} with params: {params}")

        target_function = self.function_map.get(function_name)
        if not target_function:
            return AgentResult(False, f"NurseAgent does not have a function named '{function_name}'.")

        # --- REFINED LOGIC ---
        is_list_request = user_input and any(keyword in user_input.lower() for keyword in ["list", "who are", "show wards"])
        display_intent = user_input and any(keyword in user_input.lower() for keyword in ["plot", "table", "chart", "graph", "show me", "display"])

        try:
            result_data = target_function(params)
            
            if "error" in result_data:
                return AgentResult(False, result_data["error"])

            # 1. Handle simple list requests directly to avoid verbose LLM summaries
            if is_list_request:
                self.logger.log(f"Direct list intent detected for '{function_name}'. Formatting directly.")
                summary = ""
                if function_name == "get_all_patients_info" and "ward" in user_input.lower():
                    wards = list(set(result_data.get('ward_name', [])))
                    summary = "Here are the wards: \n- " + "\n- ".join(wards)
                elif function_name == "get_all_patients_info":
                     patient_list = [f"{name} (ID: {pid})" for name, pid in zip(result_data.get('patients', []), result_data.get('patient_id', []))]
                     summary = "Here are the patients: \n- " + "\n- ".join(patient_list)
                if summary:
                    return AgentResult(True, summary)

            # 2. Handle requests that require plotting or tables
            if display_intent:
                self.logger.log(f"Display intent detected for '{function_name}'. Returning raw data.")
                plot_type = None
                if function_name == "get_patient_info":
                    plot_type = "single_patient_dashboard"
                elif function_name == "get_all_patients_info":
                    plot_type = "all_patients_table"
                elif function_name == "get_patient_vitals_trend":
                    if "table" in user_input.lower():
                        plot_type = "vitals_table"
                    else:
                        plot_type = "vitals_trend"
                
                if plot_type:
                    summary = "Here is the requested data."
                    return AgentResult(True, summary, plot_data=result_data, plot_type=plot_type)
            
            # 3. Default behavior: summarize with LLM for complex queries
            prompt = f"The user asked: '{user_input}'. The system retrieved the following data: {json.dumps(result_data, indent=2)}. Please provide a concise, human-readable summary of this information."
            summary = self.llm.invoke(prompt)
            return AgentResult(True, summary)
            
        except Exception as e:
            self.logger.log(f"Error executing {function_name}: {e}")
            return AgentResult(False, f"An error occurred while executing the task: {e}")
