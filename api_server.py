import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from orchestrator import build_demo_system
from utils.logging_config import AppLogger
from tools.rag_tool import AdvancedRAGTool
from langchain_community.llms import Ollama

# --- Data Models for API Requests and Responses ---
class QueryRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    response_text: str
    routed_to: list[str]
    plot_data: dict | None = None
    plot_type: str | None = None

# --- Global State Management ---
state = {}

# --- Lifespan Management (for model loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once on server startup to initialize the agentic system.
    """
    print("INFO:     Initializing the Clinical Intelligence System...")
    app_logger = AppLogger()
    
    # Read model and host configurations from environment variables
    general_model = os.getenv("GENERAL_MODEL", "qwen2:7b")
    med_model = os.getenv("MED_MODEL", "meditron:7b")
    # Add the RAG LLM model from environment variables
    rag_model = os.getenv("RAG_MODEL", "qwen2:7b") 
    ollama_host = os.getenv("OLLAMA_HOST", "localhost")
    ollama_base_url = f"http://{ollama_host}:11434"
    
    rag_tool_instance = None 

    # Correctly call build_demo_system with all required arguments
    state["orchestrator"] = build_demo_system(
        general_model_name=general_model,
        med_model_name=med_model,
        rag_llm_name=rag_model, # Pass the RAG model
        rag_tool_instance=rag_tool_instance,
        logger=app_logger,
        ollama_base_url=ollama_base_url # Pass the Ollama URL
    )
    print("INFO:     System Initialized and ready to accept requests.")
    yield
    # This code runs on shutdown
    state.clear()
    print("INFO:     System shut down.")


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)


# --- API Endpoint Definition ---
@app.post("/predict", response_model=AgentResponse)
async def predict(request: QueryRequest):
    """
    Accepts a user query, processes it through the agentic framework,
    and returns the consolidated response.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        orchestrator = state.get("orchestrator")
        if not orchestrator:
            raise HTTPException(status_code=503, detail="System is not initialized. Please wait.")

        result = orchestrator.handle(request.query)

        return AgentResponse(
            response_text=result.text,
            routed_to=result.metadata.get('routed_to', []),
            plot_data=result.plot_data,
            plot_type=result.plot_type
        )
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# To run this server directly (for testing):
# uvicorn api_server:app --reload

