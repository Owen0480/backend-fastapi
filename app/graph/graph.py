"""
LangGraph를 사용한 여행 추천 시스템 그래프 구성
"""
from langgraph.graph import StateGraph, END
from app.graph.state import TravelState
from app.graph.nodes import (
    user_input_node,
    intent_classifier_node,
    travel_classifier_llm_node,
    travel_info_extractor_node,
    missing_info_check_node,
    clarifying_question_node,
    retrieval_node,
    recommendation_node,
    post_action_node,
    add_favorite_node,
    favorite_list_node
)


def route_intent(state: TravelState) -> str:
    """
    Intent에 따라 다음 노드로 라우팅합니다.
    """
    intent = state.get("intent")
    
    if intent == "recommend_place" or intent == "recommend_accommodation":
        return "travel_flow"
    elif intent == "add_favorite":
        return "add_favorite"
    elif intent == "show_favorites":
        return "show_favorites"
    elif intent == "plan_trip":
        return "plan_trip"
    else:
        return "travel_flow"  # 기본값


def route_missing_info(state: TravelState) -> str:
    """
    정보 부족 여부에 따라 다음 노드로 라우팅합니다.
    """
    missing_info = state.get("missing_info", [])
    
    if missing_info:
        return "clarifying"
    else:
        return "retrieval"


def create_travel_graph() -> StateGraph:
    """
    여행 추천 시스템의 전체 그래프를 생성합니다.
    """
    # StateGraph 생성
    workflow = StateGraph(TravelState)
    
    # ==================== 노드 추가 ====================
    # Entry 노드
    workflow.add_node("user_input", user_input_node)
    
    # Intent 분류
    workflow.add_node("intent_classifier", intent_classifier_node)
    
    # 여행지/숙소 추천 Flow (LLM 분류 → 정보 추출 → …)
    workflow.add_node("travel_classifier_llm", travel_classifier_llm_node)
    workflow.add_node("info_extractor", travel_info_extractor_node)
    workflow.add_node("missing_info_check", missing_info_check_node)
    workflow.add_node("clarifying", clarifying_question_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("post_action", post_action_node)
    
    # 찜 관련 노드
    workflow.add_node("add_favorite", add_favorite_node)
    workflow.add_node("show_favorites", favorite_list_node)
    
    # ==================== 엣지 추가 ====================
    # Entry → Intent 분류
    workflow.set_entry_point("user_input")
    workflow.add_edge("user_input", "intent_classifier")
    
    # Intent 분류 → 조건 분기
    workflow.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "travel_flow": "travel_classifier_llm",
            "add_favorite": "add_favorite",
            "show_favorites": "show_favorites",
            "plan_trip": "travel_classifier_llm"  # 일정 만들기도 여행지 추천과 동일 플로우
        }
    )
    
    # 여행지 추천 Flow: LLM 분류 → 정보 추출 → 부족 여부 판단
    workflow.add_edge("travel_classifier_llm", "info_extractor")
    workflow.add_edge("info_extractor", "missing_info_check")
    
    # 정보 부족 여부에 따른 분기
    workflow.add_conditional_edges(
        "missing_info_check",
        route_missing_info,
        {
            "clarifying": "clarifying",
            "retrieval": "retrieval"
        }
    )
    
    # 질문 생성 → 다시 사용자 입력으로 (Loop)
    workflow.add_edge("clarifying", END)  # 실제로는 다시 user_input으로 돌아가야 하지만, 
                                          # 외부에서 새로운 입력을 받아 다시 그래프를 실행해야 함
    
    # RAG 검색 → 추천 생성 → 후속 액션
    workflow.add_edge("retrieval", "recommendation")
    workflow.add_edge("recommendation", "post_action")
    workflow.add_edge("post_action", END)
    
    # 찜 추가 → 종료
    workflow.add_edge("add_favorite", END)
    
    # 찜 목록 조회 → 종료
    workflow.add_edge("show_favorites", END)
    
    # 그래프 컴파일
    return workflow.compile()


# lazy: import 시점에 생성하지 않고 첫 사용 시 생성 (reload 시 asyncio teardown 예외 완화)
_travel_graph = None


def get_travel_graph():
    """첫 호출 시 그래프를 생성해 반환합니다. reload 시 teardown 예외를 줄이기 위해 lazy 로드."""
    global _travel_graph
    if _travel_graph is None:
        _travel_graph = create_travel_graph()
    return _travel_graph


# 하위 호환용 (get_travel_graph 사용 권장)
travel_graph = None
