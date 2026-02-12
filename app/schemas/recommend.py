"""
이미지 추천 API 응답 규격 (recommend_service 응답과 동일한 Key 사용)
프론트엔드 UI에서 place_name, guide, image_file 등 동일 키로 사용
"""
from pydantic import BaseModel
from typing import Optional, List


class RecommendResultItem(BaseModel):
    """results 배열 한 항목. 이미지는 image_url(전체 URL) 사용."""
    place_name: str
    address: str
    score: float
    image_file: str  # 파일명 (하위 호환)
    image_url: str  # 이미지 전체 URL (프론트는 이걸 사용)
    guide: str


class RecommendAnalyzeResponse(BaseModel):
    """POST /api/v1/recommend/analyze 응답 스키마"""
    success: bool
    count: Optional[int] = None
    results: Optional[List[RecommendResultItem]] = None
    ai_analysis: Optional[str] = None
    message: Optional[str] = None

    class Config:
        extra = "allow"
