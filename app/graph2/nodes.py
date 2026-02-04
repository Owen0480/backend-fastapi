from app.graph2.state import AgentState

def info_extractor(state: AgentState) -> AgentState:
    """사용자의 입력을 분석하여 컨텍스트를 추출하는 로직"""
    # 초기 상태값 설정 (데이터가 없을 경우)
    if "user_context" not in state:
        state["user_context"] = {}
    if "info_complete" not in state:
        state["info_complete"] = False
    
    # TODO: 실제 LLM 로직 연동
    # 현재는 테스트를 위해 메시지 내용에 '추천'이 포함되면 완료된 것으로 간주하는 예시 로직
    last_message = state["messages"][-1]
    content = last_message.get("content", "")
    
    if "추천" in content:
        state["info_complete"] = True
    
    return state

def travel_planner(state: AgentState) -> AgentState:
    """추출된 컨텍스트를 기반으로 여행 계획을 생성하는 로직"""
    # 초기 상태값 설정
    if "recommendations" not in state:
        state["recommendations"] = ""
    
    # TODO: 실제 로직 연동
    state["recommendations"] = "사용자님의 취향에 맞는 여행 코스를 추천해 드립니다..."
    state["is_valid"] = True
    
    return state

def recommendation_validator(state: AgentState) -> AgentState:
    """생성된 추천안의 정합성을 검증하는 로직"""
    # 초기 상태값 설정
    if "is_valid" not in state:
        state["is_valid"] = True
    
    # 검증 통과한 것으로 간주
    state["is_valid"] = True
    state["feedback"] = "검증 완료"
    
    return state