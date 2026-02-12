from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    """
    Request model for document classification.
    Field name matches the case study spec: 'document_text'.
    """
    document_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The document text to classify. Max length 5000 characters."
    )


class PredictionResponse(BaseModel):
    """
    Response model for classification result.
    Includes spec-required fields (message, label) plus operational metadata.
    """
    message: str = Field(..., description="Human-readable status message.")
    label: str = Field(..., description="Predicted category (e.g., 'sport', 'technology', 'other').")
    confidence: float = Field(..., description="The model's confidence score (0.0 to 1.0).")
    is_ood: bool = Field(False, description="True if the prediction was overridden to 'other' due to low confidence.")
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds.")
    model_version: str = Field(..., description="Version of the model used.")
