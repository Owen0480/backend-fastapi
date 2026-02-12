"""
여행 추천 그래프 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.graph.service import travel_graph_service

router = APIRouter()


class TravelRequest(BaseModel):
    """여행 추천 요청 (대화 턴 누적용)"""
    message: str
    user_id: Optional[int] = None
    conversation_history: Optional[List[dict]] = None  # 이전 턴 [{"role":"user"|"assistant","content":"..."}, ...]
    previous_filters: Optional[Dict[str, Any]] = None  # 이전까지 쌓인 정보(region, theme, who, why 등)


class TravelResponse(BaseModel):
    """여행 추천 응답 (누적 filters로 누구와/어디로/테마 등 함께 표출)"""
    response: str
    intent: Optional[str] = None
    who: Optional[str] = None
    why: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    when_info: Optional[Dict[str, Any]] = None
    conversation_stage: Optional[str] = None
    recommendations: List[dict] = []
    clarifying_question: Optional[str] = None
    post_actions: List[str] = []
    favorite_items: List[dict] = []
    needs_clarification: bool = False
    filters: Optional[Dict[str, Any]] = None  # 누적: region, theme, who, why 등
    error: Optional[str] = None


@router.post("/travel", response_model=TravelResponse)
async def process_travel_request(request: TravelRequest):
    """
    사용자 입력을 처리하고 여행 추천 그래프를 실행합니다.
    
    예시:
    ```json
    {
        "message": "부산 바다 보이는 감성 숙소 추천해줘",
        "user_id": 123
    }
    ```
    """
    try:
        result = await travel_graph_service.process_user_input(
            message=request.message,
            user_id=request.user_id,
            conversation_history=request.conversation_history,
            previous_filters=request.previous_filters,
        )
        
        return TravelResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/travel/graph")
async def get_graph_structure():
    """
    그래프 구조를 시각화합니다. (디버깅용)
    """
    try:
        mermaid_diagram = travel_graph_service.get_graph_visualization()
        return {"mermaid": mermaid_diagram}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
