from app.graph2.graph import app_graph

async def run_travel_chat(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    # 새로운 메시지만 전달하여 LangGraph 상태 업데이트
    input_data = {
        "messages": [{"role": "user", "content": user_input}]
    }

    # 그래프 실행 (ainvoke 사용)
    final_state = await app_graph.ainvoke(input_data, config=config)

    # 응답 추출
    # info_complete가 True면 recommendations를, 아니면 마지막 메시지 내용을 반환
    if final_state.get("info_complete"):
        answer = final_state.get("recommendations", "추천 결과가 없습니다.")
    else:
        # 마지막 메시지 (보통 AI의 추가 질문) 추출
        messages = final_state.get("messages", [])
        answer = messages[-1]["content"] if messages else "응답을 생성할 수 없습니다."

    return {
        "answer": answer,
        "thread_id": thread_id,
        "info_complete": bool(final_state.get("info_complete", False))
    }