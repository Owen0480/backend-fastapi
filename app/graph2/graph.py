from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph2.state import AgentState
from app.graph2.nodes import info_extractor, travel_planner, recommendation_validator

# 엣지 결정을 위한 조건부 함수
def decide_if_info_enough(state: AgentState):
    if state["info_complete"]:
        return "plan"
    return "ask_more"

def decide_if_valid(state: AgentState):
    if state["is_valid"]:
        return "valid"
    return "invalid"

def create_graph():
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("extractor", info_extractor)
    workflow.add_node("planner", travel_planner)
    workflow.add_node("validator", recommendation_validator)

    # 엔트리 포인트 설정
    workflow.set_entry_point("extractor")

    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        "extractor",
        decide_if_info_enough,
        {
            "plan": "planner",
            "ask_more": END
        }
    )

    workflow.add_edge("planner", "validator")

    workflow.add_conditional_edges(
        "validator",
        decide_if_valid,
        {
            "valid": END,
            "invalid": "planner"
        }
    )

    # 메모리 세이버 추가하여 thread_id 기반 상태 유지 지원
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

app_graph = create_graph()