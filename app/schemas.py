from pydantic import BaseModel, Field

class DocumentRequest(BaseModel):
    """
    Request model for document classification.
    """
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000, 
        description="The legal document text to classify. Max length 5000 characters."
    )

class PredictionResponse(BaseModel):
    """
    Response model for classification result.
    """
    category: str = Field(..., description="Predicted category (e.g., 'contract', 'court-ruling', 'other').")
    confidence: float = Field(..., description="The model's confidence score (0.0 to 1.0).")
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds.")
    model_version: str = Field(..., description="Version of the model used.")
