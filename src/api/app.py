"""
FastAPI application for text summarization and generation inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inference_optimizer import OptimizedInference


# Initialize FastAPI app
app = FastAPI(
    title="Domain-Specific Text Generation & Summarization API",
    description="API for summarizing and generating domain-specific text using fine-tuned LLM",
    version="1.0.0"
)


# Request models
class SummarizeRequest(BaseModel):
    """Request model for summarization endpoint."""
    text: str = Field(..., description="Text to summarize", min_length=10)
    max_length: int = Field(150, description="Maximum length of summary", ge=50, le=500)
    num_beams: int = Field(4, description="Number of beams for beam search", ge=1, le=10)


class SummarizeBatchRequest(BaseModel):
    """Request model for batch summarization endpoint."""
    texts: List[str] = Field(..., description="List of texts to summarize", min_items=1, max_items=10)
    max_length: int = Field(150, description="Maximum length of summaries", ge=50, le=500)
    num_beams: int = Field(4, description="Number of beams for beam search", ge=1, le=10)


class GenerateRequest(BaseModel):
    """Request model for text generation endpoint."""
    prompt: str = Field(..., description="Prompt for text generation", min_length=5)
    max_length: int = Field(200, description="Maximum length of generated text", ge=50, le=1000)
    num_beams: int = Field(4, description="Number of beams for beam search", ge=1, le=10)
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.1, le=2.0)


# Response models
class SummarizeResponse(BaseModel):
    """Response model for summarization endpoint."""
    summary: str
    input_length: int
    summary_length: int


class SummarizeBatchResponse(BaseModel):
    """Response model for batch summarization endpoint."""
    summaries: List[str]
    count: int


class GenerateResponse(BaseModel):
    """Response model for text generation endpoint."""
    generated_text: str
    prompt_length: int
    generated_length: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    model_loaded: bool
    device: str


# Global variable to hold the model
inference_engine: Optional[OptimizedInference] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global inference_engine
    
    # Get model path from environment variable or use default
    model_path = os.getenv("MODEL_PATH", "models/checkpoints/final_model")
    use_4bit = os.getenv("USE_4BIT", "false").lower() == "true"
    use_8bit = os.getenv("USE_8BIT", "false").lower() == "true"
    
    print(f"Loading model from {model_path}")
    print(f"4-bit quantization: {use_4bit}")
    print(f"8-bit quantization: {use_8bit}")
    
    try:
        inference_engine = OptimizedInference(
            model_path=model_path,
            use_4bit=use_4bit,
            use_8bit=use_8bit
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but inference endpoints will not work.")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="running",
        model_loaded=inference_engine is not None,
        device=inference_engine.device if inference_engine else "none"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthResponse(
        status="running",
        model_loaded=inference_engine is not None,
        device=inference_engine.device if inference_engine else "none"
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarize a long text.
    
    Args:
        request: SummarizeRequest with text and parameters
        
    Returns:
        SummarizeResponse with generated summary
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        summary = inference_engine.summarize(
            text=request.text,
            max_length=request.max_length,
            num_beams=request.num_beams
        )
        
        return SummarizeResponse(
            summary=summary,
            input_length=len(request.text),
            summary_length=len(summary)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")


@app.post("/summarize-batch", response_model=SummarizeBatchResponse)
async def summarize_batch(request: SummarizeBatchRequest):
    """
    Summarize multiple texts in a batch (more efficient).
    
    Args:
        request: SummarizeBatchRequest with texts and parameters
        
    Returns:
        SummarizeBatchResponse with generated summaries
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        summaries = inference_engine.summarize_batch(
            texts=request.texts,
            max_length=request.max_length,
            num_beams=request.num_beams
        )
        
        return SummarizeBatchResponse(
            summaries=summaries,
            count=len(summaries)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch summarization: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate domain-specific text based on a prompt.
    
    Args:
        request: GenerateRequest with prompt and parameters
        
    Returns:
        GenerateResponse with generated text
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_text = inference_engine.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            num_beams=request.num_beams,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt_length=len(request.prompt),
            generated_length=len(generated_text)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during text generation: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
