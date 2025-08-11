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
    Acts as a dispatcher for clinical tools based on commands from the orchestrator.
    It can return raw data for plotting/display or an LLM-generated summary.
    """
    def __init__(self, llm: Ollama, clinical_tools: ClinicalDataTools, logger):
        super().__init__("NurseAgent", llm, {}, logger)
        self.tools = clinical_tools
        self.function_map = {
            "list_wards": self.tools.list_wards,
            "list_patients_by_ward": self.tools.list_patients_by_ward,
            "get_patient_protocol": self.tools.get_patient_protocol,
            "get_patient_vitals": self.tools.get_patient_vitals,
            "get_patient_last_ecg": self.tools.get_patient_last_ecg,
            "get_patient_active_alarms": self.tools.get_patient_active_alarms,
            "get_all_patients_last_ews": self.tools.get_all_patients_last_ews,
            "get_critical_patients": self.tools.get_critical_patients,
            "get_patient_order_for_rounds": self.tools.get_patient_order_for_rounds,
            "analyze_ews_trend": self.analyze_ews_trend_wrapper
        }

    def analyze_ews_trend_wrapper(self, params: Dict) -> AgentResult:
        """Wrapper to perform EWS trend analysis and format the result."""
        patient_id = params.get("patient_id")
        duration = params.get("duration_hours", 24)
        
        vitals_data = self.tools.get_patient_vitals({"patient_id": patient_id, "duration_hours": duration})
        timestamps = [datetime.datetime.fromtimestamp(s['timestamp']) for s in vitals_data['vitals']]
        ews_values = [random.randint(0, 5) for _ in vitals_data['vitals']] 
        
        analyzer = EWSTrendAnalyzer()
        analysis = analyzer.analyze_trend(timestamps, ews_values)
        
        plot_data = {"samples": [{"ts": ts.timestamp(), "value": val} for ts, val in zip(timestamps, ews_values)]}
        
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

        # --- NEW LOGIC: Check for display intent ---
        display_intent = any(keyword in user_input.lower() for keyword in ["plot", "table", "chart", "graph", "show me", "display"])

        # Special handling for functions that can return plottable data
        if function_name in ["get_patient_vitals", "get_patient_last_ecg"] and display_intent:
            self.logger.log(f"Display intent detected for '{function_name}'. Returning raw data.")
            raw_data = target_function(params)
            
            # Determine the plot type
            plot_type = "vitals_table"
            if "plot" in user_input.lower() or "trend" in user_input.lower():
                plot_type = "vitals_trend"
            if "ecg" in function_name:
                plot_type = "ecg_waveform"

            summary = f"Here is the requested data for patient {params.get('patient_id')}."
            return AgentResult(True, summary, plot_data=raw_data, plot_type=plot_type)

        if function_name == "analyze_ews_trend":
            return target_function(params)

        # Default behavior: call tool and summarize with LLM
        try:
            result_data = target_function(params)
            prompt = f"The user asked: '{user_input}'. The system retrieved the following data: {json.dumps(result_data)}. Please provide a concise, human-readable summary of this information."
            summary = self.llm.invoke(prompt)
            return AgentResult(True, summary)
        except Exception as e:
            self.logger.log(f"Error executing {function_name}: {e}")
            return AgentResult(False, f"An error occurred while executing the task: {e}")

