# routing_config.py
# This file centralizes the keywords and sources used for agent routing and tools.

ROUTING_KEYWORDS = {
    # "Search" is now a keyword for both WebAgent and RAGAgent
    "webagent": [
        "search for", "latest on", "find information on", "what is the news on", "google", "search"
    ],
    "medagent": [
        "usmle", "clinical research", "medical guideline", "pubmed", "treatment for",
        "what is the protocol", "diagnose", "symptoms of", "medical question",
        "evidence-based", "what does the literature say"
    ],
    "nurse": [
        "hr trend", "heart rate", "ecg", "ews", "early warning", "mews", "alarm", 
        "plot", "waveform", "vitals", "monitoring data",
        "patient improvement", "patient deterioration"
    ],
    "rag": [
        "how to", "specification", "manual", "arrhythmia", "device", "sensor", 
        "place this", "technical specs", "search the knowledge base", "pdf", "search"
    ],
    "emr": [
        # Patient-Specific Queries
        "demographic", "contact details", "insurance", "admission history", "adt",
        "location", "emergency contact", "medical history", "prescriptions",
        "allergy", "progress notes", "discharge summary", "discharge report", 
        "past visit", "lab results", "laboratory test", "radiology report", 
        "imaging studies", "pathology report", "biopsy", "diagnostic procedure",
        "treatment plan", "medication administration", "mar", "nursing assessment",
        "surgical report", "therapy record",
        
        # Disease and Population-Level Queries
        "disease prevalence", "outbreak detection", "mortality statistics", 
        "length of stay",
        
        # Clinical Decision Support
        "treatment protocols", "drug interaction", "clinical guidelines",
        
        # Quality and Performance
        "patient safety", "readmission rates", "infection control",
        
        # Research and Analytics
        "cohort identification", "outcomes research", "clinical trial",
        
        # Operational Analytics
        "resource utilization", "bed management", "workflow optimization"
    ]
}

# Configurable web sources for the WebAgent
WEB_AGENT_SOURCES = {
    "pubmed": "https://pubmed.ncbi.nlm.nih.gov/",
    "mayo_clinic": "https://www.mayoclinic.org/search/search-results",
    "webmd": "https://www.webmd.com/search/search_results/default.aspx"
}

