from typing import Annotated, TypedDict
import operator

class AgentState(TypedDict):
    """여행 추천 에이전트의 상태를 관리하는 객체"""

    # 사용자의 선호도 및 제약 사항 (예: 장소, 예산, 동행인 등)
    user_preferences: dict
    
    # 검색된 여행지 후보 목록
    candidates: list
    
    # 추가 정보가 결합된 여행 데이터 (RAG, API 결과 등)
    enriched_data: list
    
    # 필터링 및 랭킹이 적용된 선택지 목록
    filtered_options: list
    
    # 사용자에게 제공할 최종 추천안 리스트
    final_recommendations: list
    
    # 현재 단계의 재시도 횟수
    retry_count: int
    enrich_retry_count: int
    
    # 허용되는 최대 재시도 횟수
    max_retries: int
    
    # 결과물의 유효성 검사 점수 (0.0 ~ 1.0)
    validation_score: float
    
    # 유효성 검사 결과에 대한 상세 피드백 내용
    validation_feedback: str
    
    # 수집된 사용자 정보의 품질 점수 (필수 정보 충족 여부 등)
    info_quality_score: float
    
    # 대화 기록 (기존 히스토리에 새 메시지가 누적되도록 operator.add 사용)
    messages: Annotated[list, operator.add]
    
    # 현재 대화의 의도 (recommend_travel, general_chat, irrelevant_chat)
    intent: str
    
    # 현재 진행 중인 프로세스 단계 (예: extractor, planner, validator 등)
    current_step: str