# routing_config.py
# This file centralizes sources for the WebAgent.
# Keyword-based routing is now handled by the LLM-based orchestrator.

# Configurable web sources for the WebAgent's "search" function
WEB_AGENT_SOURCES = {
    "pubmed": "https://pubmed.ncbi.nlm.nih.gov/",
    "mayo_clinic": "https://www.mayoclinic.org/search/search-results",
    "webmd": "https://www.webmd.com/search/search_results/default.aspx"
}

# Configurable web sources for the WebAgent's "scrape" function
WEB_AGENT_SCRAPE_SOURCES = {
    "cdc_news": "https://www.cdc.gov/media/releases/index.html",
    "who_news": "https://www.who.int/news"
}

