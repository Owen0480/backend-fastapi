from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage


class TravelState(TypedDict):
    """여행 추천 시스템의 상태를 관리하는 State"""
    # 사용자 입력 관련
    user_input: List[BaseMessage]  # 사용자의 메시지 히스토리
    latest_message: str  # 가장 최근 사용자 입력
    
    # 의도 분류
    intent: Optional[str]  # recommend_place, recommend_accommodation, add_favorite, show_favorites, plan_trip
    
    # LLM 분류 결과 (여행지 추천용)
    who: Optional[str]  # 페르소나/동행자: family_with_kids, couple, parents_trip, solo
    why: Optional[str]  # 목적/테마: relaxation, activity, culture, food
    constraints: Dict[str, Any]  # 제약/선호: transport(car/public), budget(value/luxury), pet_friendly
    when_info: Dict[str, Any]  # 시기: season, operating_hours 등
    conversation_stage: Optional[str]  # exploration, refinement, confirmation
    
    # 정보 추출 결과
    filters: Dict[str, Any]  # 지역, 테마, 기간, 계절, 예산 등 필터 조건
    missing_info: List[str]  # 부족한 정보 목록 (who/why/constraints/when 포함)
    
    # RAG 및 추천 결과
    retrieved_docs: List[Dict[str, Any]]  # 벡터 DB에서 검색된 문서들
    recommendations: List[Dict[str, Any]]  # 최종 추천 결과
    
    # 사용자 정보
    user_id: Optional[int]  # 로그인한 사용자 ID
    
    # 질문 및 응답
    clarifying_question: Optional[str]  # 추가 정보 요청 질문
    response: Optional[str]  # 최종 응답 메시지
    
    # 후속 액션
    post_actions: List[str]  # 제안할 후속 액션들 (예: "지도 보기", "찜 추가")
    
    # 찜 관련
    favorite_items: List[Dict[str, Any]]  # 찜 목록 조회 결과
