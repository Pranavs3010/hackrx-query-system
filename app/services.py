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
    LOADER_MAP = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.eml': UnstructuredEmailLoader,
    }

    # --- MODIFIED AS REQUESTED ---
    # The default model for the LLM has been changed to "gemini-pro".
    # WARNING: This will cause the '404 model not found' error to return,
    # leading to a 500 Internal Server Error. The working value is "gemini-1.5-flash-latest".
    def __init__(self, embedding_device='cpu', model_name="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="gemini-2.5-flash", chunk_size=1000, chunk_overlap=150):
    # --- END OF CHANGE ---
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': embedding_device}
        )

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.0,
            convert_system_message_to_human=True
        )

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
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

        self.vector_store = None

    def process_document(self, doc_url: str):
        logger.info("Step 1: Document Processing Started")
        try:
            doc_path = download_document(doc_url)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Failed to download the document: {e}")

        file_extension = doc_path.suffix.lower()
        loader_cls = self.LOADER_MAP.get(file_extension)
        if not loader_cls:
            doc_path.unlink(missing_ok=True)
            raise ValueError(f"Unsupported file type: '{file_extension}'.")

        try:
            loader = loader_cls(str(doc_path))
            documents = loader.load()
        finally:
            doc_path.unlink(missing_ok=True)


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)

        logger.info(f"Document split into {len(chunks)} chunks.")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        logger.info("FAISS vector store created successfully.")
        logger.info("Step 1: Document Processing Finished")

    def answer_question(self, question: str) -> str:
        if not self.vector_store:
            raise RuntimeError("Document has not been processed. Call process_document() first.")

        logger.info(f"Searching for context for question: '{question}'")
        retrieved_chunks = self.vector_store.similarity_search(question, k=5)

        if not retrieved_chunks:
            return "The information is not available in the provided document."

        context = "\n\n---\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        llm_chain = self.prompt | self.llm

        logger.info("Invoking LLM to generate the final answer.")
        response = llm_chain.invoke({"context": context, "question": question})

        return response.content