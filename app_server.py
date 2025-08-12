import streamlit as st
import pandas as pd
import requests
import os
from plotting import plot_agent_result
from agents.base_agent import AgentResult

def main():
    st.set_page_config(page_title="Clinical Intelligence System", layout="wide")

    # --- Title and Help Section ---
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.title("🩺 Clinical Intelligence System")
    with col2:
        st.write("##") 
        with st.popover("❔ Help"):
            st.markdown("""
            ### Clinical Intelligence System
            This is a sophisticated, multi-agent framework designed to assist healthcare professionals by providing quick and accurate answers to a wide range of clinical, technical, and research-related queries.
            
            At its core is an **Orchestrator** that intelligently analyzes a user's natural language query. It identifies the user's intent and routes the request to one or more specialized agents, each with a distinct area of expertise. If multiple agents are needed, the orchestrator consolidates their findings into a single, coherent response.
            
            ---
            
            ### The Specialized Agents
            * **Nurse Agent 📈:** The frontline expert for real-time patient data.
            * **EMR Agent 📂:** The digital records clerk for historical patient documents.
            * **RAG Agent 📄:** The expert for information contained within uploaded documents.
            * **Web Agent 🌐:** For the most up-to-date, external information.
            * **MedAgent 🧑‍⚕️:** The general medical consultant using a specialized LLM.
            """)

    # --- Sidebar ---
    with st.sidebar:
        st.info("System configuration is managed by the backend API server.")
        st.markdown(
            '<div style="margin-top: 2em;"><a href="http://www.coherentix.com" target="_blank" style="font-size: 12px; color: grey; text-decoration: none;">Coherentix Labs</a></div>',
            unsafe_allow_html=True
        )

    # --- Main Interface ---
    st.subheader("Enter Your Query")
    user_query = st.text_area("e.g., 'Show me the vitals for patient CARD101 in a table'", height=100)

    if st.button("Submit Query"):
        if not user_query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Processing..."):
                try:
                    # The API host is read from an environment variable in a Docker deployment
                    # Default to localhost for local testing
                    api_host = os.getenv("API_HOST", "127.0.0.1")
                    api_url = f"http://{api_host}:8000/predict"
                    
                    # Make the POST request to the FastAPI backend
                    response = requests.post(api_url, json={"query": user_query})
                    response.raise_for_status()
                    
                    api_result = response.json()
                    
                    st.subheader("Agent Response")
                    routed_to_list = api_result.get('routed_to', ['N/A'])
                    routed_to_str = ', '.join([name.title() for name in routed_to_list])
                    st.markdown(f"**Agent(s) Queried: `{routed_to_str}`**")
                    
                    st.markdown(api_result.get("response_text"))

                    # --- UPDATED DISPLAY LOGIC ---
                    if api_result.get("plot_data"):
                        # Create a mock AgentResult object to pass to the plotter
                        mock_result = AgentResult(
                            success=True,
                            text="",
                            plot_data=api_result["plot_data"],
                            plot_type=api_result["plot_type"]
                        )
                        # The plot_agent_result function now handles rendering directly
                        plot_agent_result(mock_result, st)
                    
                except requests.exceptions.ConnectionError:
                     st.error(f"Could not connect to the API server at {api_url}. Please ensure the backend service is running.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

