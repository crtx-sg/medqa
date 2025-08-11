import re
from typing import Dict, Any, Optional, Callable
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from utils.logging_config import AppLogger
from routing_config import ROUTING_KEYWORDS

# --- EMR Agent Tools ---

def tool_emr_fetch(params: Dict[str, Any], logger: AppLogger) -> Dict[str, Any]:
    """Mocks fetching a document from an EMR system based on document type."""
    patient_id = params.get("patient_id", "UnknownPatient")
    doc_type = params.get("document_type", "Unknown Document")
    logger.log(f"TOOL: tool_emr_fetch called for {patient_id}, doc: {doc_type}")
    
    # Generate mock content based on the document type
    content = f"--- {doc_type.title()} for Patient {patient_id} ---\n"
    if "discharge" in doc_type.lower():
        content += "Diagnosis: Stable. Condition: Good. Follow-up in 2 weeks."
    elif "lab" in doc_type.lower():
        content += "WBC: 10.5, HGB: 14.1, PLT: 250k. All values within normal limits."
    elif "radiology" in doc_type.lower():
        content += "Chest X-Ray: No acute cardiopulmonary process identified."
    elif "medication" in doc_type.lower():
        content += "Current prescriptions: Lisinopril 10mg daily, Metformin 500mg twice daily."
    else:
        content += "No specific information available for this document type in the mock system."
        
    return {"patient_id": patient_id, "document_type": doc_type, "content": content}

def get_emr_tools(logger: AppLogger) -> Dict[str, Callable]:
    return {"emr": lambda p: tool_emr_fetch(p, logger)}

# --- EMR Agent Definition ---

class EMRAgent(BaseAgent):
    """Handles queries for retrieving patient records from the EMR."""
    def __init__(self, llm: Ollama, tools: Dict[str, Callable], logger: AppLogger):
        super().__init__("EMRAgent", llm, tools, logger)

    def _extract_document_type(self, user_input: str) -> str:
        """Intelligently determines the requested document type from the query."""
        ui = user_input.lower()
        # Check for specific document types using the routing keywords
        emr_keywords = ROUTING_KEYWORDS.get("emr", [])
        for keyword in emr_keywords:
            if keyword in ui:
                # Return a formatted version of the keyword
                return keyword.replace("_", " ").title()
        return "General Inquiry"

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        self.logger.log(f"{self.name} is handling: '{user_input}'")
        ctx = context or {}
        match = re.search(r"patient\s*([A-Za-z0-9_\-]+)", user_input, re.I)
        patient = match.group(1) if match else ctx.get("patient_id", "C")
        
        doc_type = self._extract_document_type(user_input)
        
        if doc_type != "General Inquiry":
            self.logger.log(f"Identified document type: '{doc_type}'")
            doc = self.tools["emr"]({"patient_id": patient, "document_type": doc_type})
            prompt = f"Briefly summarize the following EMR document for patient {patient}:\n\n{doc['content']}"
            llm_resp = self.llm.invoke(prompt)
            return AgentResult(True, llm_resp, metadata=doc)
            
        return AgentResult(False, "EMR Agent could not determine a specific document type to fetch from your query.")

