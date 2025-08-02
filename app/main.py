# /intelligent-query-retrieval-system/app/main.py

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from .api_models import QueryRequest, QueryResponse
from .services import QueryProcessor

# Load API keys and other environment variables from the .env file
load_dotenv()

# --- Application and Security Setup ---
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="Processes documents to answer contextual questions using LLMs.",
    version="1.0.0"
)

API_KEY = os.getenv("API_AUTH_TOKEN")
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Dependency function to validate the API bearer token."""
    if not api_key or api_key != f"Bearer {API_KEY}":
        logger.warning("Failed authentication attempt.")
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# --- Service Initialization ---
# Check for the Google API key on startup
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

# Instantiate the QueryProcessor. It will be reused for all requests.
query_processor = QueryProcessor()

# --- API Endpoint Definition ---
@app.post(
    "/api/v1/hackrx/run", 
    response_model=QueryResponse,
    tags=["Query Retrieval"],
    summary="Process a document and answer questions"
)
async def run_submission(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    """
    This endpoint implements the full RAG pipeline:
    1.  **Input Document**: Downloads a document from the provided URL.
    2.  **Embedding & Indexing**: Processes and indexes the document in-memory using FAISS.
    3.  **Retrieval & Answering**: For each question, retrieves relevant context
        and uses the Gemini LLM to generate a precise answer.
    4.  **JSON Output**: Returns a structured JSON response with all answers.
    """
    try:
        # This is a stateful operation; the processor holds the indexed document.
        query_processor.process_document(request.documents)

        answers = []
        for question in request.questions:
            answer = query_processor.answer_question(question)
            answers.append(answer)
        
        logger.info("Successfully answered all questions.")
        return QueryResponse(answers=answers)

    except ValueError as e:
        logger.error(f"Validation Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) # Bad Request for unsupported file type
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Intelligent Query–Retrieval System is running."}