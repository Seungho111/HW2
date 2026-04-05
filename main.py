from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import tutor_model

app = FastAPI(
    title="Chinese Grammar AI Tutor API",
    description="A lightweight AI tutor for Chinese grammar, ready for MLOps pipelines.",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>API is running but index.html is missing.</h1>")

@app.post("/chat", response_model=ChatResponse)
def chat_with_tutor(request: ChatRequest):
    if tutor_model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable or failed to load.")
    
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
        reply = tutor_model.generate_response(request.message)
        return ChatResponse(reply=reply)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
