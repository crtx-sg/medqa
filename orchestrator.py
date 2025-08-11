from typing import Tuple, Dict, Any, Optional
from langchain_community.llms import Ollama
from agents.base_agent import BaseAgent, AgentResult
from agents.nurse_agent import NurseAgent, get_nurse_tools
from agents.rag_agent import RAGAgent
from agents.emr_agent import EMRAgent, get_emr_tools
from agents.med_agent import MedAgent, get_med_tools
from agents.web_agent import WebAgent, get_web_tools
from utils.logging_config import AppLogger
from utils.rag_tool import AdvancedRAGTool
from routing_config import ROUTING_KEYWORDS

class Orchestrator:
    """
    Handles user requests by classifying intent, routing to one or more agents,
    and synthesizing responses if necessary.
    """
    def __init__(self, llm: Ollama, agents: Dict[str, BaseAgent], logger: AppLogger):
        self.llm = llm
        self.agents = agents
        self.logger = logger

    def intent_classify(self, user_input: str) -> list[str]:
        ui = user_input.lower()
        self.logger.log(f"Classifying intent for: '{ui}'")
        
        matched_agents = []
        for agent_name, keywords in ROUTING_KEYWORDS.items():
            if any(k in ui for k in keywords):
                matched_agents.append(agent_name)
        
        if not matched_agents:
            self.logger.log("No specific keywords matched. Defaulting to 'medagent'.")
            return ["medagent"]
        
        self.logger.log(f"Intent classified, relevant agents: {matched_agents}")
        return matched_agents

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        agent_keys = self.intent_classify(user_input)
        
        if len(agent_keys) == 1:
            agent_key = agent_keys[0]
            agent = self.agents.get(agent_key)
            if not agent:
                return AgentResult(False, f"Error: No agent found for key '{agent_key}'")
            
            self.logger.log(f"Routing to single agent: {agent.name}")
            result = agent.handle(user_input, context)
            result.metadata.update({"routed_to": [agent_key]})
            return result
        else:
            self.logger.log(f"Multi-agent workflow triggered for: {agent_keys}")
            consolidated_responses = []
            final_plot_data = None
            final_plot_type = None

            for agent_key in agent_keys:
                agent = self.agents.get(agent_key)
                if agent:
                    self.logger.log(f"Querying agent: {agent.name}")
                    result = agent.handle(user_input, context)
                    if result.success:
                        response_header = f"--- Response from {agent.name} ---\n"
                        consolidated_responses.append(response_header + result.text)
                        if result.plot_data and not final_plot_data:
                            final_plot_data = result.plot_data
                            final_plot_type = result.plot_type
            
            if not consolidated_responses:
                return AgentResult(False, "None of the relevant agents could provide a response.")

            synthesis_prompt = f"""
            You are a master orchestrator AI. Your job is to synthesize information from multiple specialized agents into a single, comprehensive response for a clinician.
            The original user query was: "{user_input}"
            The following agents provided information:
            {''.join(consolidated_responses)}
            Please consolidate these responses into one final, coherent answer that directly addresses the user's query.
            """
            self.logger.log("Synthesizing responses from multiple agents...")
            final_response_text = self.llm.invoke(synthesis_prompt)
            
            return AgentResult(
                success=True,
                text=final_response_text,
                plot_data=final_plot_data,
                plot_type=final_plot_type,
                metadata={"routed_to": agent_keys}
            )

def build_demo_system(
    general_model_name: str, 
    med_model_name: str, 
    rag_tool_instance: Optional[AdvancedRAGTool],
    logger: AppLogger,
    ollama_base_url: str # Add parameter for Ollama URL
) -> Orchestrator:
    logger.log(f"Building system with General Model: {general_model_name}, Medical Model: {med_model_name}, Ollama URL: {ollama_base_url}")
    
    # Initialize LLM clients with the provided base URL
    general_llm = Ollama(model=general_model_name, base_url=ollama_base_url)
    medical_llm = Ollama(model=med_model_name, base_url=ollama_base_url)

    agents = {
        "nurse": NurseAgent(general_llm, get_nurse_tools(logger), logger),
        "rag": RAGAgent(general_llm, rag_tool_instance, logger),
        "emr": EMRAgent(general_llm, get_emr_tools(logger), logger),
        "medagent": MedAgent(medical_llm, get_med_tools(logger), logger),
        "webagent": WebAgent(general_llm, get_web_tools(logger), logger)
    }
    
    logger.log("System build complete.")
    return Orchestrator(general_llm, agents, logger)

