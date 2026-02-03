"""
여행 스타일 컨트롤러 (domain 기준)
"""
from fastapi import APIRouter, HTTPException

from app.domain.travel_style import INTEREST_OPTIONS
from app.schemas.travel_style import InterestRequest
from app.services.travel_style import analyze_interests

router = APIRouter()


@router.get("/options")
def get_options():
    """domain INTEREST_OPTIONS 반환 (프론트/스프링 통일용)"""
    return {"options": INTEREST_OPTIONS}


@router.post("/analyze")
async def analyze_travel_type(request: InterestRequest):
    """
    관심사 3개를 받아 여행 타입을 LLM으로 분석합니다.
    """
    if len(request.interests) != 3:
        raise HTTPException(status_code=400, detail="관심사는 정확히 3개 선택해야 합니다.")

    try:
        result = analyze_interests(request.interests)
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
