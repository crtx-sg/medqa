# config.py

# Models for general-purpose agents (Nurse, RAG, EMR, Web)
AVAILABLE_MODELS = {
    "qwen2:7b": "Qwen2 7B",
    "phi3:latest": "Phi-3",
    "llama3:2b": "Llama3.2 1B",
    "deepseek-r1:1.5b": "DeepSeek R1 1.5B"
}

# Specialized models for the MedAgent
MED_AGENT_MODELS = {
    # Ollama Models
    "meditron:7b": "Meditron",
    "biomistral": "Ollama - BioMistral",
    "medllama2": "Ollama - MedLlama2",
    # Conceptual Non-Ollama Models (add API logic to use them)
    "clinical_camel": "API - Clinical Camel (Conceptual)",
    "medgemma": "API - MedGemma (Conceptual)",
    "openbiollm": "API - OpenBioLLM (Conceptual)"
}

# --- RAG System Configuration ---
SUPPORTED_VECTOR_DBS = ["qdrant", "milvus"]
SUPPORTED_EMBEDDINGS = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"]

