import os
import uuid
import pymupdf
import pytesseract
from PIL import Image
import io
from langchain_community.vectorstores import Qdrant, Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import qdrant_client
from pymilvus import connections, utility
from utils.logging_config import AppLogger

class AdvancedRAGTool:
    """
    A class that encapsulates the logic for a configurable RAG system,
    handling PDF loading, chunking, embedding, and querying.
    """
    def __init__(self, llm, embedding_model_name, vector_db_name, db_config, logger: AppLogger):
        self.logger = logger
        self.logger.log(f"Initializing RAG tool with DB: {vector_db_name}, Embedding: {embedding_model_name}")
        self.llm = llm
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_db_name = vector_db_name
        self.db_config = db_config
        self.vector_store = self._setup_vector_store()

    def _setup_vector_store(self):
        """Sets up the vector database based on configuration."""
        self.logger.log(f"Setting up vector store: {self.vector_db_name}")
        try:
            if self.vector_db_name == "qdrant":
                client = qdrant_client.QdrantClient(url=self.db_config.get("qdrant_url", ":memory:"))
                return Qdrant(client=client, collection_name="rag_collection", embeddings=self.embedding_model)
            
            elif self.vector_db_name == "milvus":
                connection_args = {
                    "host": self.db_config.get("milvus_host"),
                    "port": self.db_config.get("milvus_port")
                }
                connections.connect(alias="default", **connection_args)
                return Milvus(embedding_function=self.embedding_model, connection_args=connection_args, collection_name="rag_collection")
            
            self.logger.log(f"Vector store '{self.vector_db_name}' set up successfully.")
        except Exception as e:
            self.logger.log(f"ERROR setting up vector store: {e}")
            raise
        return None

    def add_pdf_to_knowledge_base(self, pdf_file):
        """Loads and processes a PDF, adding its content to the vector store."""
        if not self.vector_store:
            self.logger.log("ERROR: Vector store not initialized.")
            return

        self.logger.log(f"Processing PDF: {pdf_file.name}")
        try:
            temp_file_path = f"temp_{uuid.uuid4().hex}.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(pdf_file.read())

            pdf_doc = pymupdf.open(temp_file_path)
            all_docs = []
            for page_num, page in enumerate(pdf_doc):
                page_text = page.get_text()
                doc = Document(page_content=page_text, metadata={"source": pdf_file.name, "page": page_num})
                all_docs.append(doc)
            
            pdf_doc.close()
            os.remove(temp_file_path)

            documents = self.text_splitter.split_documents(all_docs)
            self.logger.log(f"Adding {len(documents)} document chunks to {self.vector_db_name}.")

            # Conditional logic to handle Milvus ID requirement
            if self.vector_db_name == "milvus":
                # Milvus requires explicit IDs when auto_id is False
                doc_ids = [str(uuid.uuid4()) for _ in documents]
                self.vector_store.add_documents(documents, ids=doc_ids)
            else:
                # Qdrant handles ID generation automatically
                self.vector_store.add_documents(documents)

            self.logger.log("PDF content added to knowledge base.")
        except Exception as e:
            self.logger.log(f"ERROR loading PDF: {e}")
            raise

    def query(self, query_text: str) -> str:
        """Queries the vector store and uses the LLM to generate a response."""
        if not self.vector_store:
            return "RAG system not initialized. Please configure and load documents first."
        
        self.logger.log(f"RAG Tool: Searching for context for query: '{query_text}'")
        docs = self.vector_store.similarity_search(query_text, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if not context:
            return "No relevant information found in the knowledge base for your query."

        prompt = f"""
        You are an expert assistant. Use the following retrieved context to answer the user's query.
        If the context does not contain the answer, state that you could not find the information.

        Context:
        {context}

        Query: {query_text}

        Answer:
        """
        self.logger.log("Invoking LLM with retrieved context.")
        response = self.llm.invoke(prompt)
        return response

