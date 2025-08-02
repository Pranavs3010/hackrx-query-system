# /intelligent-query-retrieval-system/app/api_models.py

from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    """Defines the structure for the incoming API request."""
    documents: str = Field(..., description="URL to the PDF, DOCX, or EML document.")
    questions: List[str] = Field(..., description="List of natural language questions to ask about the document.")

class QueryResponse(BaseModel):
    """Defines the structure for the outgoing API response."""
    answers: List[str] = Field(..., description="List of generated answers corresponding to the questions.")