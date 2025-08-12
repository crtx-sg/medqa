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
    Acts as a dispatcher for the consolidated clinical tools based on commands
    from the orchestrator.
    """
    def __init__(self, llm: Ollama, clinical_tools: ClinicalDataTools, logger):
        super().__init__("NurseAgent", llm, {}, logger)
        self.tools = clinical_tools
        # Map the new, consolidated function names to the tool methods
        self.function_map = {
            "get_patient_info": self.tools.get_patient_info,
            "get_all_patients_info": self.tools.get_all_patients_info,
            "get_patient_alarms": self.tools.get_patient_alarms,
            "get_critical_patients": self.tools.get_critical_patients,
            "get_patient_vitals_trend": self.tools.get_patient_vitals_trend,
            "analyze_ews_trend": self.analyze_ews_trend_wrapper,
        }

    def analyze_ews_trend_wrapper(self, params: Dict) -> AgentResult:
        """Wrapper to perform EWS trend analysis over a duration and format the result."""
        patient_id = params.get("patient_id") or params.get("patient_name")
        self.logger.log(f"Analyzing EWS trend for patient {patient_id}.")

        trend_data = self.tools.get_patient_vitals_trend(params)
        if "error" in trend_data:
            return AgentResult(False, trend_data["error"])
        
        timestamps = [v['timestamp'] for v in trend_data.get("vitals", [])]
        ews_values = [v['ews_score'] for v in trend_data.get("vitals", [])]
        
        analyzer = EWSTrendAnalyzer()
        analysis = analyzer.analyze_trend([datetime.datetime.fromtimestamp(ts) for ts in timestamps], ews_values)
        
        plot_data = {"samples": [{"ts": ts, "value": val} for ts, val in zip(timestamps, ews_values)], "patient_id": patient_id}
        
        prompt = f"Summarize the following EWS trend analysis for patient {patient_id}: {analysis['clinical_interpretation']}"
        summary = self.llm.invoke(prompt)
        
        return AgentResult(True, summary, plot_data=plot_data, plot_type="ews_trend")

    def handle(self, user_input: str, context: Dict[str, Any]) -> AgentResult:
        function_name = context.get("function")
        params = context.get("params", {})
        
        self.logger.log(f"NurseAgent executing function: {function_name} with params: {params}")

        target_function = self.function_map.get(function_name)
        if not target_function:
            return AgentResult(False, f"NurseAgent does not have a function named '{function_name}'.")

        if function_name == "analyze_ews_trend":
            return target_function(params)

        display_intent = user_input and any(keyword in user_input.lower() for keyword in ["plot", "table", "chart", "graph", "show me", "display"])

        try:
            result_data = target_function(params)
            
            if "error" in result_data:
                return AgentResult(False, result_data["error"])

            if display_intent:
                plot_type = None
                # --- REFINED DISPLAY LOGIC ---
                if function_name == "get_patient_info":
                    plot_type = "single_patient_dashboard"
                elif function_name == "get_all_patients_info":
                    plot_type = "all_patients_table"
                elif function_name == "get_patient_vitals_trend":
                    # Check user intent specifically for trend data
                    if "table" in user_input.lower():
                        plot_type = "vitals_table"
                    else: # Default to a graphical plot for trends
                        plot_type = "vitals_trend"
                
                if plot_type:
                    summary = "Here is the requested data."
                    return AgentResult(True, summary, plot_data=result_data, plot_type=plot_type)
            
            # Default behavior: summarize with LLM if no specific display format is requested
            prompt = f"The user asked: '{user_input}'. The system retrieved the following data: {json.dumps(result_data, indent=2)}. Please provide a concise, human-readable summary of this information."
            summary = self.llm.invoke(prompt)
            return AgentResult(True, summary)
            
        except Exception as e:
            self.logger.log(f"Error executing {function_name}: {e}")
            return AgentResult(False, f"An error occurred while executing the task: {e}")

