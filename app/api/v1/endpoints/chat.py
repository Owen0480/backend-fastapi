from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import llm_service

router = APIRouter()

@router.post("/response", response_model=ChatResponse)
async def generate_chat_response(request: ChatRequest):
    try:
        response = await llm_service.generate_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
