from typing import Annotated, List, TypedDict, Any
import operator

# 1. State 정의: 노드 간에 공유될 상태 데이터
class AgentState(TypedDict):
    # 사용자의 전체 대화 기록 (operator.add를 사용하여 기존 리스트에 추가되도록 설정)
    messages: Annotated[List[dict], operator.add]
    
    # 추출된 사용자의 여행 조건 (예산, 장소, 기간, 취향 등)
    user_context: dict
    
    # LLM이 생성한 추천안
    recommendations: str
    
    # 검증 결과 (True/False 및 피드백 메시지)
    is_valid: bool
    feedback: str
    
    # 필수 정보가 모두 수집되었는지 여부
    info_complete: bool