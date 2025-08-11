import re
import time
import math
import random
import datetime
from typing import Dict, Any, Optional, Callable
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger
from utils.ews_analyzer import EWSTrendAnalyzer # Import the new analyzer

# --- Nurse Agent Tools ---

def tool_fetch_vitals(params: Dict[str, Any], logger: AppLogger) -> Dict[str, Any]:
    """Generates synthetic vital sign data for trend plotting."""
    patient_id = params.get("patient_id", "UnknownPatient")
    vital = params.get("vital_name", "HR")
    duration_hours = params.get("duration_hours", 24)
    logger.log(f"TOOL: tool_fetch_vitals called for {patient_id}, vital: {vital}, duration: {duration_hours}h")

    now = int(time.time())
    data = []
    num_points = duration_hours * 4 
    
    for i in range(num_points):
        timestamp = now - (num_points - i) * 15 * 60
        if vital == "HR":
            value = 70 + random.randint(-5, 5) + (i % 7)
        elif vital == "EWS":
            # Simulate a more dynamic EWS score for better trend analysis
            base = 3
            trend = -math.sin(i / (num_points/4)) # Simulate a wave
            noise = random.uniform(-1, 1)
            value = max(0, int(base + trend * 2 + noise))
            value = min(10, value)
        else:
            value = 0
        data.append({"ts": timestamp, "value": value})
        
    return {"patient_id": patient_id, "vital": vital, "samples": data}

def tool_fetch_ecg_waveform(params: Dict[str, Any], logger: AppLogger) -> Dict[str, Any]:
    """Generates a synthetic ECG waveform."""
    patient_id = params.get("patient_id", "UnknownPatient")
    logger.log(f"TOOL: tool_fetch_ecg_waveform called for {patient_id}")

    seconds = 12
    fs = 250
    n = seconds * fs
    t = [i / fs for i in range(n)]
    lead_signal = [math.sin(2 * math.pi * 1.0 * tt) * (1 + 0.05 * random.uniform(-1, 1)) for tt in t]
    return {"patient_id": patient_id, "lead": "II", "fs": fs, "t": t, "signal": lead_signal}

def tool_get_latest_ews(params: Dict[str, Any], logger: AppLogger) -> Dict[str, Any]:
    """Simulates retrieving the single most recent MEWS score for a patient."""
    patient_id = params.get("patient_id", "UnknownPatient")
    logger.log(f"TOOL: tool_get_latest_ews called for {patient_id}")
    
    score = random.randint(0, 8)
    timestamp = int(time.time()) - random.randint(60, 300) 
    
    return {"patient_id": patient_id, "ews_score": score, "timestamp": timestamp}

def tool_analyze_ews_trend(params: Dict[str, Any], logger: AppLogger) -> Dict[str, Any]:
    """
    Analyzes the EWS trend by fetching data and processing it with EWSTrendAnalyzer.
    """
    logger.log("TOOL: tool_analyze_ews_trend called.")
    # Step 1: Fetch the raw EWS data
    vitals_data = tool_fetch_vitals(params, logger)
    samples = vitals_data.get("samples", [])
    if not samples:
        return {"error": "No EWS data found to analyze."}

    timestamps = [datetime.datetime.fromtimestamp(s['ts']) for s in samples]
    ews_values = [s['value'] for s in samples]

    # Step 2: Analyze the data with the new tool
    analyzer = EWSTrendAnalyzer()
    analysis_results = analyzer.analyze_trend(timestamps, ews_values)
    
    # Log the specific reasoning from the analyzer
    logger.log(f"EWS Analysis Complete. Reason: {analysis_results.get('analysis_reasoning', 'N/A')}")
    
    # Combine raw data with analysis for plotting and response generation
    analysis_results['raw_data'] = vitals_data
    return analysis_results


def get_nurse_tools(logger: AppLogger) -> Dict[str, Callable]:
    """Factory to create tool dictionary with logger context."""
    return {
        "vitals_trend": lambda p: tool_fetch_vitals(p, logger),
        "ecg_waveform": lambda p: tool_fetch_ecg_waveform(p, logger),
        "latest_ews": lambda p: tool_get_latest_ews(p, logger),
        "analyze_ews_trend": lambda p: tool_analyze_ews_trend(p, logger),
    }

# --- Nurse Agent Definition ---

class NurseAgent(BaseAgent):
    """Handles clinical monitoring queries."""
    def __init__(self, llm: Ollama, tools: Dict[str, Callable], logger: AppLogger):
        super().__init__("NurseAgent", llm, tools, logger)

    def _extract_patient(self, user_input: str, context: Dict) -> str:
        match = re.search(r"patient\s*([A-Za-z0-9_\-]+)", user_input, re.I)
        return match.group(1) if match else context.get("patient_id", "A")

    def _extract_duration(self, user_input: str, default: int = 24) -> int:
        match = re.search(r"(\d+)\s*hours", user_input, re.I)
        return int(match.group(1)) if match else default

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        self.logger.log(f"{self.name} is handling: '{user_input}'")
        ctx = context or {}
        patient = self._extract_patient(user_input, ctx)
        ui_lower = user_input.lower()

        # Specific query for the latest EWS/MEWS score (non-trending)
        if ("latest" in ui_lower or "what is the" in ui_lower) and ("ews" in ui_lower or "mews" in ui_lower):
            res = self.tools["latest_ews"]({"patient_id": patient})
            score = res.get('ews_score')
            ts = datetime.datetime.fromtimestamp(res.get('timestamp')).strftime('%Y-%m-%d %H:%M:%S')
            text_response = f"The latest MEWS score for patient {patient} is **{score}**, recorded at {ts}."
            return AgentResult(True, text_response)

        # Plotting HR trend
        if "hr" in ui_lower or "heart rate" in ui_lower:
            duration = self._extract_duration(ui_lower, 48)
            res = self.tools["vitals_trend"]({"patient_id": patient, "vital_name": "HR", "duration_hours": duration})
            prompt = f"Summarize the provided HR data for patient {patient} over the last {duration} hours."
            llm_resp = self.llm.invoke(prompt)
            return AgentResult(True, llm_resp, plot_data=res, plot_type="hr_trend")

        # Analyzing and Plotting EWS/MEWS trend
        if "ews" in ui_lower or "mews" in ui_lower:
            duration = self._extract_duration(ui_lower, 24)
            # Use the new analysis tool
            res = self.tools["analyze_ews_trend"]({
                "patient_id": patient, 
                "vital_name": "EWS", 
                "duration_hours": duration
            })
            
            if "error" in res:
                return AgentResult(False, res["error"])

            # Create a prompt for the LLM to summarize the structured analysis, including the reasoning
            prompt = f"""
            You are a clinical assistant summarizing a patient's status.
            Based on the following EWS trend analysis for patient {patient}, provide a concise summary.
            
            Analysis Results:
            - Patient Status: {res.get('patient_status')}
            - Confidence in Trend: {res.get('confidence')}
            - Recent Improvement Noted: {res.get('improvement')}
            - Clinical Recommendation: {res.get('clinical_interpretation')}
            - Basis for Analysis: {res.get('analysis_reasoning')}
            
            Combine this into a brief, easy-to-read summary for a clinician.
            """
            llm_resp = self.llm.invoke(prompt)
            return AgentResult(True, llm_resp, plot_data=res.get('raw_data'), plot_type="ews_trend")

        # Getting ECG waveform
        if "ecg" in ui_lower or "waveform" in ui_lower:
            res = self.tools["ecg_waveform"]({"patient_id": patient})
            prompt = f"Provide a brief, one-sentence technical description of a 12-second ECG Lead II waveform."
            llm_resp = self.llm.invoke(prompt)
            return AgentResult(True, llm_resp, plot_data=res, plot_type="ecg_waveform")
            
        # Listing alarms
        if "alarm" in ui_lower:
            alarms_text = (f"Recent alarms for {patient}:\n"
                           f"1. [High HR @ 20:55] - HR: 130bpm, SpO2: 98%\n"
                           f"2. [Low SpO2 @ 19:10] - HR: 95bpm, SpO2: 89%")
            return AgentResult(True, alarms_text)

        return AgentResult(False, "NurseAgent could not match the request to a supported operation.")

