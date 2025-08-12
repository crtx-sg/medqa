import streamlit as st
import pandas as pd
from config import (
    AVAILABLE_MODELS, MED_AGENT_MODELS, 
    SUPPORTED_VECTOR_DBS, SUPPORTED_EMBEDDINGS
)
from orchestrator import build_demo_system
from plotting import plot_agent_result
from utils.logging_config import AppLogger
from tools.rag_tool import AdvancedRAGTool
from langchain_community.llms import Ollama
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

    # Initialize session state
    if 'rag_tool' not in st.session_state:
        st.session_state.rag_tool = None
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

    # --- Sidebar ---
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=True):
            st.header("Agent Configuration")
            general_model_selection = st.selectbox(
                "Choose a General Purpose Model", 
                options=list(AVAILABLE_MODELS.keys()), 
                format_func=lambda x: AVAILABLE_MODELS[x]
            )
            
            st.subheader("Medical Agent")
            med_model_selection = st.selectbox(
                "Choose a Specialist Model", 
                options=list(MED_AGENT_MODELS.keys()), 
                format_func=lambda x: MED_AGENT_MODELS[x]
            )
            
            st.divider()

            st.header("RAG System Configuration")
            rag_llm_model = st.selectbox(
                "Select RAG LLM", 
                options=list(AVAILABLE_MODELS.keys()), 
                format_func=lambda x: AVAILABLE_MODELS[x]
            )
            vector_db = st.selectbox(
                "Select Vector Database", 
                SUPPORTED_VECTOR_DBS, 
                index=SUPPORTED_VECTOR_DBS.index("milvus") if "milvus" in SUPPORTED_VECTOR_DBS else 0
            )
            embedding_model = st.selectbox("Select Embedding Model", SUPPORTED_EMBEDDINGS)
            db_config = {}
            if vector_db == "qdrant":
                db_config["qdrant_url"] = st.text_input("Qdrant URL", ":memory:")
            elif vector_db == "milvus":
                db_config["milvus_host"] = st.text_input("Milvus Host", "localhost")
                db_config["milvus_port"] = st.text_input("Milvus Port", "19530")

            uploaded_files = st.file_uploader("Upload PDFs to Knowledge Base", type=["pdf"], accept_multiple_files=True)

            if st.button("Initialize System"):
                if not uploaded_files:
                    st.warning("Please upload at least one PDF to initialize the RAG system.")
                else:
                    with st.spinner("Initializing System and loading documents..."):
                        try:
                            app_logger = AppLogger()
                            rag_llm = Ollama(model=rag_llm_model)
                            rag_tool = AdvancedRAGTool(
                                llm=rag_llm,
                                embedding_model_name=embedding_model,
                                vector_db_name=vector_db,
                                db_config=db_config,
                                logger=app_logger
                            )
                            for pdf in uploaded_files:
                                rag_tool.add_pdf_to_knowledge_base(pdf)
                            
                            st.session_state.rag_tool = rag_tool
                            st.session_state.system_initialized = True
                            st.success("System Initialized!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Initialization Failed: {e}")
        
        if st.session_state.system_initialized:
            st.success("✅ System Initialized")
        
        st.markdown(
            '<div style="margin-top: 2em;"><a href="http://www.coherentix.com" target="_blank" style="font-size: 12px; color: grey; text-decoration: none;">Coherentix Labs</a></div>',
            unsafe_allow_html=True
        )

    # --- Main Interface ---
    st.subheader("Enter Your Query")
    user_query = st.text_area("e.g., 'Show me the vitals for patient CARD101 in a table'", height=100)

    if st.button("Submit Query"):
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system using the settings in the sidebar first.")
        elif not user_query:
            st.warning("Please enter a query.")
        else:
            app_logger = AppLogger()
            with st.spinner("Processing..."):
                try:
                    orchestrator = build_demo_system(
                        general_model_name=general_model_selection,
                        med_model_name=med_model_selection,
                        rag_llm_name=rag_llm_model,
                        rag_tool_instance=st.session_state.rag_tool,
                        logger=app_logger,
                        ollama_base_url="http://localhost:11434"
                    )
                    
                    result = orchestrator.handle(user_query)
                    
                    st.subheader("Agent Response")
                    routed_to_list = result.metadata.get('routed_to', ['N/A'])
                    routed_to_str = ', '.join([name.title() for name in routed_to_list])
                    st.markdown(f"**Agent(s) Queried: `{routed_to_str}`**")
                    
                    st.markdown(result.text)

                    # --- UPDATED DISPLAY LOGIC ---
                    if result.plot_data:
                        # The plot_agent_result function now handles rendering directly
                        plot_agent_result(result, st)
                    
                    with st.expander("Show Processing Logs"):
                        st.text_area("", "".join(app_logger.get_logs()), height=200)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    app_logger.log(f"ERROR: {e}")
                    with st.expander("Show Error Logs"):
                        st.text_area("", "".join(app_logger.get_logs()), height=200)

if __name__ == "__main__":
    main()

