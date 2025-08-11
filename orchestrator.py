import json
from typing import List, Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from agents.nurse_agent import NurseAgent
from agents.rag_agent import RAGAgent
from agents.emr_agent import EMRAgent
from agents.med_agent import MedAgent
from agents.web_agent import WebAgent
from utils.logging_config import AppLogger
from tools.rag_tool import AdvancedRAGTool
from tools.clinical_tools import ClinicalDataTools

class Orchestrator:
    """
    Handles user requests by using an LLM to classify intent and route to the
    appropriate agent and tool.
    """
    def __init__(self, llm: Ollama, agents: Dict[str, BaseAgent], logger: AppLogger):
        self.llm = llm
        self.agents = agents
        self.logger = logger
        self._define_tools()

    def _define_tools(self):
        """Creates a structured definition of all available tools for the LLM router."""
        self.tool_definitions = [
            # Nurse Agent Tools
            {"agent": "nurse", "function": "list_wards", "description": "Lists all hospital care units or wards. Use when the user asks to see all wards."},
            {"agent": "nurse", "function": "list_patients_by_ward", "description": "Lists all patients in a specific ward. Requires a 'ward' parameter (e.g., 'Cardiology'). Use when the user wants to see patients in a particular unit."},
            {"agent": "nurse", "function": "get_patient_protocol", "description": "Retrieves the treatment protocol (prescriptions, diet, allergies) for a specific patient. Requires a 'patient_id' parameter."},
            {"agent": "nurse", "function": "get_patient_vitals", "description": "Retrieves a patient's vital signs over a specified period. Requires 'patient_id' and 'duration_hours' parameters."},
            {"agent": "nurse", "function": "get_patient_last_ecg", "description": "Gets the most recent 12-second ECG strip for a patient. Requires a 'patient_id' parameter."},
            {"agent": "nurse", "function": "get_patient_active_alarms", "description": "Gets all active clinical alarms for a specific patient. Requires a 'patient_id' parameter."},
            {"agent": "nurse", "function": "get_all_patients_last_ews", "description": "Retrieves the last updated Early Warning Score for all patients."},
            {"agent": "nurse", "function": "get_critical_patients", "description": "Lists all patients whose EWS trend indicates deterioration."},
            {"agent": "nurse", "function": "get_patient_order_for_rounds", "description": "Provides the list of patients in a ward, ordered by criticality. Requires a 'ward' parameter."},
            {"agent": "nurse", "function": "analyze_ews_trend", "description": "Performs a trend analysis on a patient's EWS scores. Use for queries about 'patient improvement' or 'deterioration'. Requires 'patient_id' and 'duration_hours'."},
            
            # EMR Agent Tools
            {"agent": "emr", "function": "get_patient_image_study", "description": "Retrieves a medical imaging study (like an X-ray) from the PACS system for a patient. Requires a 'patient_id' parameter."},
            
            # RAG Agent Tool
            {"agent": "rag", "function": "query_knowledge_base", "description": "Searches the local knowledge base of uploaded PDF documents to answer a query. Use for questions about manuals, specifications, or content within provided files."},
            
            # Web Agent Tools
            {"agent": "webagent", "function": "scrape_predefined_links", "description": "Scrapes a predefined list of websites for the latest general news or information."},
            {"agent": "webagent", "function": "search_predefined_links", "description": "Searches a predefined list of websites to answer a specific query."},

            # Med Agent (General Knowledge)
            {"agent": "medagent", "function": "answer_medical_question", "description": "Answers general medical questions, explains clinical guidelines, or tackles USMLE-style problems. Use for any query that doesn't fit other tools."}
        ]

    def intent_classify_with_llm(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Uses an LLM to classify intent and extract parameters."""
        
        prompt = f"""
        You are an intelligent hospital operations router. Your task is to analyze a user's query and determine which tool should be used to answer it.
        
        Here are the available tools:
        {json.dumps(self.tool_definitions, indent=2)}

        User Query: "{user_input}"

        Based on the query, which is the single best tool to use?
        Respond ONLY with a valid JSON object containing the chosen 'agent', the 'function', and any necessary 'parameters'.
        For example: {{"agent": "nurse", "function": "list_patients_by_ward", "parameters": {{"ward": "Cardiology"}}}}
        If a parameter like 'patient_id' is mentioned as a name (e.g., "John Doe"), use the name.
        """
        
        self.logger.log("Using LLM for intent classification...")
        response_str = self.llm.invoke(prompt)
        
        try:
            # Clean the response string to ensure it's valid JSON
            clean_response = response_str.strip().replace("```json", "").replace("```", "")
            response_json = json.loads(clean_response)
            self.logger.log(f"LLM routing decision: {response_json}")
            return response_json
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.log(f"LLM routing failed or returned invalid JSON: {e}\nResponse: {response_str}")
            return None # Fallback

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Routes the request to the appropriate agent and tool using LLM-based classification.
        """
        routing_decision = self.intent_classify_with_llm(user_input)
        
        if not routing_decision:
            return AgentResult(False, "I could not understand the request. Please try rephrasing.")

        agent_key = routing_decision.get("agent")
        function_name = routing_decision.get("function")
        params = routing_decision.get("parameters", {})

        agent = self.agents.get(agent_key)
        if not agent:
            return AgentResult(False, f"Error: No agent found for key '{agent_key}'")

        self.logger.log(f"Routing to agent: {agent.name}, function: {function_name}, params: {params}")
        
        # The agent's handle method will now need to execute the specified function
        result = agent.handle(user_input, {"function": function_name, "params": params})
        result.metadata.update({"routed_to": [agent_key]})
        return result


def build_demo_system(
    general_model_name: str, 
    med_model_name: str, 
    rag_llm_name: str,
    rag_tool_instance: Optional[AdvancedRAGTool],
    logger: AppLogger,
    ollama_base_url: str
) -> Orchestrator:
    logger.log(f"Building system with General: {general_model_name}, Medical: {med_model_name}, RAG: {rag_llm_name}")
    
    # Initialize LLM clients
    general_llm = Ollama(model=general_model_name, base_url=ollama_base_url)
    medical_llm = Ollama(model=med_model_name, base_url=ollama_base_url)
    rag_llm = Ollama(model=rag_llm_name, base_url=ollama_base_url)

    # Initialize toolsets
    clinical_tools = ClinicalDataTools(logger)

    agents = {
        "nurse": NurseAgent(general_llm, clinical_tools, logger),
        "rag": RAGAgent(rag_llm, rag_tool_instance, logger),
        "emr": EMRAgent(general_llm, clinical_tools, logger),
        "medagent": MedAgent(medical_llm, {}, logger),
        "webagent": WebAgent(general_llm, logger)
    }
    
    logger.log("System build complete.")
    # The orchestrator uses the general_llm for routing decisions
    return Orchestrator(general_llm, agents, logger)

