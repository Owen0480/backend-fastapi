"""
LangGraph 실행 서비스
그래프 실행에 타임아웃을 두고, 예외 시 에러 메시지를 응답에 포함합니다.
conversation_history로 이전 턴을 넘기면 누구와/어디로 등이 쌓여 다음 답변에 반영됩니다.
"""
import asyncio
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.graph.state import TravelState
from app.graph.graph import get_travel_graph


def _normalize_conversation_history(raw: Optional[list]) -> List[BaseMessage]:
    """conversation_history(role/content 리스트)를 HumanMessage/AIMessage 리스트로 변환."""
    if not raw:
        return []
    out: List[BaseMessage] = []
    for m in raw:
        if hasattr(m, "content") and hasattr(m, "type"):
            out.append(m)
            continue
        d = m if isinstance(m, dict) else {}
        role = (d.get("role") or d.get("type") or "user").lower()
        content = d.get("content") or d.get("text") or ""
        if role in ("user", "human", "humanmessage"):
            out.append(HumanMessage(content=str(content)))
        else:
            out.append(AIMessage(content=str(content)))
    return out


def _graph_timeout_sec() -> int:
    try:
        from app.core.config import settings
        return max(15, getattr(settings, "GRAPH_TIMEOUT_SEC", 60))
    except Exception:
        return 60


class TravelGraphService:
    """
    여행 추천 그래프를 실행하는 서비스 (타임아웃·에러 표출 적용)
    """
    
    def __init__(self):
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            self._graph = get_travel_graph()
        return self._graph
    
    async def process_user_input(
        self,
        message: str,
        user_id: Optional[int] = None,
        conversation_history: Optional[list] = None,
        previous_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        사용자 입력을 처리하고 그래프를 실행합니다.
        GRAPH_TIMEOUT_SEC 초과 시 중단하고 error 필드와 함께 응답합니다.
        """
        timeout_sec = _graph_timeout_sec()
        print("=" * 50)
        print("[Travel Graph] 실행 시작")
        print(f"[Travel Graph] 입력 메시지: {message[:80]}{'...' if len(message) > 80 else ''}")
        print(f"[Travel Graph] 그래프 타임아웃: {timeout_sec}초")
        try:
            from app.graph.nodes import _get_google_api_key
            key_ok = bool(_get_google_api_key())
            print(f"[Travel Graph] LLM(Gemini) 사용 가능 여부: {key_ok}")
        except Exception:
            print("[Travel Graph] LLM(Gemini) 사용 가능 여부: 확인 실패")
        print("=" * 50)

        history_messages = _normalize_conversation_history(conversation_history)
        seed = dict(previous_filters or {})
        initial_state: TravelState = {
            "user_input": history_messages,
            "latest_message": message,
            "intent": None,
            "who": seed.get("who"),
            "why": seed.get("why"),
            "constraints": seed.get("constraints") or {},
            "when_info": seed.get("when_info") or {},
            "conversation_stage": seed.get("conversation_stage"),
            "filters": {k: v for k, v in seed.items() if k not in ("who", "why", "constraints", "when_info", "conversation_stage")},
            "missing_info": [],
            "retrieved_docs": [],
            "recommendations": [],
            "user_id": user_id,
            "clarifying_question": None,
            "response": None,
            "post_actions": [],
            "favorite_items": []
        }
        
        try:
            final_state = await asyncio.wait_for(
                self._get_graph().ainvoke(initial_state),
                timeout=timeout_sec
            )
            print("=" * 50)
            print("[Travel Graph] 실행 완료")
            print(f"[Travel Graph] intent: {final_state.get('intent')}, who: {final_state.get('who')}, why: {final_state.get('why')}")
            print(f"[Travel Graph] needs_clarification: {bool(final_state.get('clarifying_question'))}, recommendations: {len(final_state.get('recommendations') or [])}건")
            print("=" * 50)
            filters = dict(final_state.get("filters") or {})
            filters["who"] = final_state.get("who")
            filters["why"] = final_state.get("why")
            return {
                "response": final_state.get("response", "응답을 생성하지 못했습니다."),
                "intent": final_state.get("intent"),
                "who": final_state.get("who"),
                "why": final_state.get("why"),
                "constraints": final_state.get("constraints", {}),
                "when_info": final_state.get("when_info", {}),
                "conversation_stage": final_state.get("conversation_stage"),
                "recommendations": final_state.get("recommendations", []),
                "clarifying_question": final_state.get("clarifying_question"),
                "post_actions": final_state.get("post_actions", []),
                "favorite_items": final_state.get("favorite_items", []),
                "needs_clarification": bool(final_state.get("clarifying_question")),
                "filters": filters,
                "conversation_history": final_state.get("user_input", []),
                "error": None,
            }
        except asyncio.TimeoutError:
            err_msg = f"그래프 실행 타임아웃 ({timeout_sec}초 초과)"
            print(f"[Travel Graph] {err_msg}")
            return {
                "response": err_msg,
                "intent": None,
                "who": None,
                "why": None,
                "constraints": {},
                "when_info": {},
                "conversation_stage": None,
                "recommendations": [],
                "clarifying_question": None,
                "post_actions": [],
                "favorite_items": [],
                "needs_clarification": False,
                "filters": {},
                "conversation_history": [],
                "error": err_msg,
            }
        except Exception as e:
            err_msg = str(e)
            print(f"[Travel Graph] 실행 에러: {type(e).__name__} {err_msg}")
            return {
                "response": f"오류가 발생했습니다: {err_msg}",
                "intent": None,
                "who": None,
                "why": None,
                "constraints": {},
                "when_info": {},
                "conversation_stage": None,
                "recommendations": [],
                "clarifying_question": None,
                "post_actions": [],
                "favorite_items": [],
                "needs_clarification": False,
                "filters": {},
                "conversation_history": [],
                "error": err_msg,
            }
    
    def get_graph_visualization(self) -> str:
        """
        그래프 구조를 시각화합니다. (디버깅용)
        """
        try:
            return self._get_graph().get_graph().draw_mermaid()
        except Exception as e:
            return f"그래프 시각화 오류: {str(e)}"


# 전역 서비스 인스턴스
travel_graph_service = TravelGraphService()
