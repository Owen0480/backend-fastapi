from fastapi import APIRouter, HTTPException
from app.schemas.travel_chat import ChatRequest, ChatResponse
from app.graph2.service import run_travel_chat

router = APIRouter()

@router.post("/travel/chat", response_model=ChatResponse)
async def travel_chat_endpoint(request: ChatRequest):
    try:
        # 서비스 함수 호출 시 thread_id 전달
        result = await run_travel_chat(request.message, request.thread_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))