"""
여행 스타일 DTO (domain 기준 통일)
"""
from pydantic import BaseModel
from typing import Optional


class InterestRequest(BaseModel):
    """관심사 분석 요청 (한국어 키워드 3개)"""
    interests: list[str]


class TypeInfo(BaseModel):
    """여행 타입 상세 정보"""
    description: str
    keywords: list[str]
    destinations: Optional[list[dict]] = None


class AnalyzeResponse(BaseModel):
    """여행 타입 분석 응답"""
    success: bool = True
    analysis: dict  # matched_type, confidence, reason, secondary_type, type_info, user_interests
