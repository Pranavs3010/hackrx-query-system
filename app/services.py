# /intelligent-query-retrieval-system/app/services.py

import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredEmailLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .utils import download_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    The main engine that handles the document processing and question answering workflow.
    """
    def __init__(self):
        # Initialize a fast, local embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Initialize the Google Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.0,
            convert_system_message_to_human=True # For compatibility
        )

        # Define a strict prompt template to ensure answers are based only on the document
        self.prompt_template = """
        You are an expert assistant for insurance, legal, and compliance.
        Answer the following question based ONLY on the provided context.
        If the information to answer the question is not in the context, state:
        "The information is not available in the provided document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )
        self.vector_store = None

    def process_document(self, doc_url: str):
        """
        Downloads, loads, chunks, and indexes a document from a URL.
        It automatically detects the file type (PDF, DOCX, EML).
        """
        logger.info("Step 1: Document Processing Started")
        doc_path = download_document(doc_url)
        
        # Select the correct loader based on the file extension
        file_extension = doc_path.suffix.lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(str(doc_path))
        elif file_extension == '.docx':
            loader = Docx2txtLoader(str(doc_path))
        elif file_extension == '.eml':
            loader = UnstructuredEmailLoader(str(doc_path))
        else:
            doc_path.unlink()
            raise ValueError(f"Unsupported file type: '{file_extension}'.")

        documents = loader.load()

        # Split document into smaller chunks for efficient processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunks)} chunks.")

        # Create a FAISS vector store for fast semantic search
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        logger.info("FAISS vector store created successfully.")
        
        # Clean up the downloaded file
        doc_path.unlink()
        logger.info("Step 1: Document Processing Finished")

    def answer_question(self, question: str) -> str:
        """
        Answers a single question using the indexed document.
        """
        if not self.vector_store:
            raise RuntimeError("Document has not been processed. Call process_document() first.")

        # Step 2: Clause Retrieval and Matching
        logger.info(f"Searching for context for question: '{question}'")
        retrieved_chunks = self.vector_store.similarity_search(question, k=5)
        
        if not retrieved_chunks:
            return "The information is not available in the provided document."

        context = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        
        # Step 3: Logic Evaluation with LLM
        llm_chain = self.prompt | self.llm
        logger.info("Invoking LLM to generate the final answer.")
        response = llm_chain.invoke({"context": context, "question": question})
        
        return response.content