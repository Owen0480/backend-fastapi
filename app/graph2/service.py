from app.graph2.graph import app_graph

async def run_travel_chat(user_input: str, thread_id: str):
    """
    Spring Boot 등 외부에서 호출하는 서비스 함수.
    thread_id를 기반으로 대화 상태를 유지하며 LangGraph를 실행합니다.
    """
    config = {"configurable": {"thread_id": thread_id}}

    from langchain_core.messages import HumanMessage
    
    # 초기 상태 구성: 새로운 유저 메시지만 전달
    # MemorySaver가 thread_id를 기반으로 기존 'user_preferences' 등을 유지함
    input_data = {
        "messages": [HumanMessage(content=user_input)]
    }

    # 그래프 실행 (비동기)
    # 이미 해당 thread_id의 상태가 있다면 input_data와 병합되어 업데이트됩니다.
    final_state = await app_graph.ainvoke(input_data, config=config)

    # 그래프 시각화
    print(app_graph.get_graph().draw_mermaid())

    # 응답 추출
    # messages 리스트의 마지막 항목이 AI의 응답입니다.
    messages = final_state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        answer = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", str(last_msg))
    else:
        answer = "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."

    # 정보 수집 완료 여부 판단 (모든 선호도가 수집되었고 최종 추천 단계까지 왔는지)
    info_complete = True if final_state.get("final_recommendations") else False

    return {
        "answer": answer,
        "thread_id": thread_id,
        "info_complete": info_complete
    }