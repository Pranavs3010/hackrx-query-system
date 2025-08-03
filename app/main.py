# /intelligent-query-retrieval-system/app/main.py

import os
import logging
# --- NEW: Import the 'time' library ---
import time
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from .api_models import QueryRequest, QueryResponse
from .services import QueryProcessor

# Load environment variables
load_dotenv()

# --- Application Setup ---
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="Processes documents to answer contextual questions using LLMs.",
    version="1.0.0"
)

# CORS Middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
API_KEY = os.getenv("API_AUTH_TOKEN")
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

# --- Service Initialization ---
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
query_processor = QueryProcessor()


# --- API Endpoint ---
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
    try:
        query_processor.process_document(request.documents)

        answers = []
        # --- MODIFICATION ---
        # We will loop through the questions with an index to add a delay
        for i, question in enumerate(request.questions):
            # For the first question (index 0), don't sleep.
            # For all subsequent questions, wait 2 seconds before making the next API call.
            # This respects the API's rate limits.
            if i > 0:
                logger.info("Waiting for 2 seconds to respect API rate limits...")
                time.sleep(1)
            
            answer = query_processor.answer_question(question)
            answers.append(answer)
        # --- END OF MODIFICATION ---
        
        logger.info("Successfully answered all questions.")
        return QueryResponse(answers=answers)

    except ValueError as e:
        logger.error(f"Validation Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Intelligent Query–Retrieval System is running."}