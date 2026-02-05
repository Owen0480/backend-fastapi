from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph2.state import AgentState
from app.graph2.nodes import (
    intent_classifier_node,
    general_chat_node,
    irrelevant_chat_node,
    collect_preferences_node, 
    generate_candidates_node, 
    validate_candidates_node,
    increment_retry,
    increment_enrich_retry,
    enrich_information_node,
    validate_information_node,
    filter_options_node,
    rank_destinations_node,
    final_check_node,
    present_recommendations_node
)

# ============== 조건부 엣지 함수들 ==============

def route_by_intent(state: AgentState) -> Literal["recommend", "general", "irrelevant"]:
    """의도에 따라 다른 노드로 라우팅"""
    intent = state.get("intent", "recommend_travel")
    if intent == "general_chat":
        return "general"
    elif intent == "irrelevant_chat":
        return "irrelevant"
    return "recommend"


def needs_more_info(state: AgentState) -> Literal["ask_more", "generate"]:
    """초기 입력 충분성 확인"""
    prefs = state.get("user_preferences", {})
    required_fields = ["budget", "duration", "interests"]
    
    has_all = all(field in prefs and prefs[field] for field in required_fields)
    
    print(f"✓ 입력 충분성 체크: {'충분' if has_all else '부족'}")
    return "generate" if has_all else "ask_more"


def should_regenerate_candidates(state: AgentState) -> Literal["regenerate", "proceed", "collect_preferences"]:
    """후보 재생성 여부 결정"""
    score = state.get("validation_score", 0.0)
    retry_count = state.get("retry_count", 0)
    max_retries = 2  # 후보지 품질 재시도 한도 2회로 상향/고정
    
    print(f"✓ 후보 검증 결과: 점수={score:.2f}, 재시도={retry_count}/{max_retries}")
    
    if score < 0.7:
        if retry_count < max_retries:
            return "regenerate"
        else:
            return "collect_preferences"
    
    return "proceed"


def should_reenrich(state: AgentState) -> Literal["reenrich", "proceed"]:
    """정보 재수집 여부 - 무한루프 방지를 위해 enrich_retry_count 별도 관리"""
    quality = state.get("info_quality_score", 1.0)
    enrich_retry_count = state.get("enrich_retry_count", 0)
    max_enrich_retries = 2  # 정보 품질 재시도 한도 2회
    
    print(f"✓ 정보 품질 체크: {quality:.2f} (재시도: {enrich_retry_count}/{max_enrich_retries})")
    
    # 품질이 낮고, 아직 재시도 횟수가 남았을 때만 재수집
    if quality < 0.6 and enrich_retry_count < max_enrich_retries:
        return "reenrich"
    
    # 품질이 낮더라도 재시도 한도를 초과했으면 진행
    if quality < 0.6:
        print("⚠️ 정보 품질이 낮지만 재시도 한도 초과로 다음 단계로 진행합니다.")
        
    return "proceed"

def has_viable_options(state: AgentState) -> Literal["rank", "regenerate"]:
    """필터링 후 옵션 충분성 - 무한루프 방지"""
    options = state.get("filtered_options", [])
    retry_count = state.get("retry_count", 0)
    max_retries = 2 # 재시도 한도 2회
    
    print(f"✓ 필터링 후 옵션 수: {len(options)}개 (재시도: {retry_count}/{max_retries})")
    
    # 옵션이 부족하지만 재시도 가능
    if len(options) < 3 and retry_count < max_retries:
        return "regenerate"
    
    # 옵션이 부족하지만 재시도 한도 초과 - 현재 옵션으로 진행
    if len(options) < 3:
        print("⚠️ 옵션이 부족하지만 재시도 한도 초과로 현재 옵션으로 진행합니다.")
        return "rank"
    
    return "rank"


# ============== 그래프 구성 ==============

def create_travel_graph():
    """여행 추천 그래프 생성"""
    
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.add_node("irrelevant_chat", irrelevant_chat_node)
    workflow.add_node("collect_preferences", collect_preferences_node)
    workflow.add_node("generate_candidates", generate_candidates_node)
    workflow.add_node("validate_candidates", validate_candidates_node)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("increment_enrich_retry", increment_enrich_retry)
    workflow.add_node("enrich_information", enrich_information_node)
    workflow.add_node("validate_information", validate_information_node)
    workflow.add_node("filter_options", filter_options_node)
    workflow.add_node("rank_destinations", rank_destinations_node)
    workflow.add_node("final_check", final_check_node)
    workflow.add_node("present_recommendations", present_recommendations_node)
    
    # 엣지 구성
    workflow.set_entry_point("intent_classifier")
    
    # 0. 의도 분류 → 조건부 라우팅
    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "recommend": "collect_preferences",
            "general": "general_chat",
            "irrelevant": "irrelevant_chat"
        }
    )
    
    # 0-1. 일반 대화 및 기타 대화 → 종료
    workflow.add_edge("general_chat", END)
    workflow.add_edge("irrelevant_chat", END)
    
    # 1. 입력 수집 → 조건부
    workflow.add_conditional_edges(
        "collect_preferences",
        needs_more_info,
        {
            "ask_more": END,
            "generate": "generate_candidates"
        }
    )
    
    # 2. 후보 생성 → 검증
    workflow.add_edge("generate_candidates", "validate_candidates")
    
    # 3. 후보 검증 → 조건부
    workflow.add_conditional_edges(
        "validate_candidates",
        should_regenerate_candidates,
        {
            "regenerate": "increment_retry",
            "proceed": "enrich_information",
            "collect_preferences": "collect_preferences"
        }
    )
    
    # 3-1. 재시도 카운터 증가 → 후보 재생성
    workflow.add_edge("increment_retry", "generate_candidates")
    
    # 4. 정보 수집 → 정보 검증
    workflow.add_edge("enrich_information", "validate_information")
    
    # 5. 정보 검증 → 조건부
    workflow.add_conditional_edges(
        "validate_information",
        should_reenrich,
        {
            "reenrich": "increment_enrich_retry",
            "proceed": "filter_options"
        }
    )
    
    # 5-1. 정보 보강 카운터 증가 → 정보 보강 재배행
    workflow.add_edge("increment_enrich_retry", "enrich_information")
    
    # 6. 필터링 → 조건부
    workflow.add_conditional_edges(
        "filter_options",
        has_viable_options,
        {
            "regenerate": "increment_retry",
            "rank": "rank_destinations"
        }
    )
    
    # 6-1. 재시도 카운터 증가 → 후보 재생성
    workflow.add_edge("increment_retry", "generate_candidates")
    
    # 7. 순위화 → 최종 검증
    workflow.add_edge("rank_destinations", "final_check")
    
    # 8. 최종 검증 → 결과 제시
    workflow.add_edge("final_check", "present_recommendations")
    
    # 9. 결과 제시 → 종료
    workflow.add_edge("present_recommendations", END)
    
    # 메모리 세이버 추가
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# 그래프 인스턴스 생성 및 익스포트
app_graph = create_travel_graph()