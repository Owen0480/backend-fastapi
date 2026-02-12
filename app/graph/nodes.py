"""
LangGraph 노드 구현
각 노드는 TravelState를 입력받아 수정된 상태를 반환합니다.
LLM·외부 호출에는 타임아웃을 두고, 예외 시 fallback 및 에러 로그를 남깁니다.
"""
import asyncio
import json
import os
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.graph.state import TravelState


def _llm_timeout_sec() -> int:
    try:
        from app.core.config import settings
        return max(10, getattr(settings, "LLM_TIMEOUT_SEC", 25))
    except Exception:
        return 25


def _get_google_api_key() -> Optional[str]:
    try:
        from app.core.config import settings
        key = getattr(settings, "GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    except Exception:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return (key or "").strip() or None


def _get_travel_llm():
    """API 키가 있을 때만 Gemini LLM 인스턴스 반환. 없거나 패키지 미설치 시 None."""
    key = _get_google_api_key()
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=0)
    except Exception:
        return None


def _fallback_classify_travel(user_message: str) -> Dict[str, Any]:
    """LLM 없을 때 키워드 기반 Who/Why/Constraints/When/대화단계 추론."""
    msg = (user_message or "").lower()
    who, why = "unknown", "unknown"
    for k, kw in [("family_with_kids", ["아이","유모차","키즈","가족"]), ("couple", ["커플","연인","둘이","데이트","신혼"]), ("parents_trip", ["부모님","효도"]), ("solo", ["나홀","혼자","혼밥","1인"])]:
        if any(w in msg for w in kw): who = k; break
    for k, kw in [("relaxation", ["힐링","휴식","휴양","감성"]), ("activity", ["서핑","등산","체험"]), ("culture", ["유적","미술관","문화"]), ("food", ["맛집","미식","로컬","카페"])]:
        if any(w in msg for w in kw): why = k; break
    constraints = {}
    if "자가용" in msg or "주차" in msg: constraints["transport"] = "car"
    elif "대중교통" in msg or "역세권" in msg: constraints["transport"] = "public"
    if "저렴" in msg or "가성비" in msg: constraints["budget"] = "value"
    elif "럭셔리" in msg or "고급" in msg: constraints["budget"] = "luxury"
    when_info = {}
    for k, kw in [("spring", ["봄","벚꽃"]), ("summer", ["여름","해수욕장"]), ("autumn", ["가을","단풍"]), ("winter", ["겨울","눈꽃"])]:
        if any(w in msg for w in kw): when_info["season"] = k; break
    stage = "refinement" if any(r in msg for r in ["부산","제주","강릉","서울"]) else "exploration"
    return {"who": who, "why": why, "constraints": constraints, "when_info": when_info, "conversation_stage": stage}


# ==================== 0. 여행지 추천용 LLM 분류 노드 (Who/Why/Constraints/When/대화단계) ====================
TRAVEL_CLASSIFIER_SYSTEM = """당신은 여행 추천 봇의 분류기입니다. 사용자 발화와 대화 맥락을 보고 아래 5가지를 반드시 JSON만 출력하세요. 다른 설명 없이 JSON 하나만 출력합니다.

1. who (페르소나/동행자): 누구와 가는지. 반드시 다음 중 하나만 사용.
   - family_with_kids: 아이 동반 가족 (유모차, 키즈존, 체험형)
   - couple: 커플/신혼부부 (분위기, 야경, 프라이빗)
   - parents_trip: 효도 여행 (걷기 편한, 한식 맛집, 온천/휴양)
   - solo: 나홀로 (혼밥, 치안, 게스트하우스 네트워킹)
   - unknown: 문맥에서 알 수 없음

2. why (여행 목적/테마): 얻고자 하는 가치. 반드시 다음 중 하나만 사용.
   - relaxation: 휴식/힐링 (호캉스, 스파, 숲길)
   - activity: 액티비티 (서핑, 스키, 등산, 테마파크)
   - culture: 문화/역사 (유적지, 미술관, 전통시장)
   - food: 미식 (SNS 핫플, 로컬 노포, 양조장)
   - unknown: 문맥에서 알 수 없음

3. constraints (제약/선호): 다음 키만 사용하고, 추론 가능한 것만 채우고 없으면 빈 객체.
   - transport: "car" | "public" (자차면 주차, 대중교통이면 역세권)
   - budget: "value" | "luxury" | null (가성비/배낭 vs 럭셔리)
   - pet_friendly: true | false | null (반려동물 동반 여부)

4. when_info (시기): 다음 키만 사용, 추론 가능한 것만.
   - season: "spring"|"summer"|"autumn"|"winter"|null (벚꽃/계곡·해수욕장/단풍/눈꽃)
   - need_night_info: true|false (야간 개장, 월요일 휴무 등 확인 필요 여부)

5. conversation_stage (대화 단계): 반드시 다음 중 하나만.
   - exploration: 아직 구체적 지역 없음 (예: "강원도 어디가 좋아?")
   - refinement: 지역은 정함, 상세 조건 필요 (예: "강릉에서 조용한 카페 찾아줘")
   - confirmation: 특정 장소 찜·예약 고려 단계

출력 형식(따옴표 등 정확히 JSON):
{"who":"solo","why":"relaxation","constraints":{"transport":"car","budget":"value","pet_friendly":null},"when_info":{"season":"summer","need_night_info":false},"conversation_stage":"exploration"}
"""


async def travel_classifier_llm_node(state: TravelState) -> TravelState:
    """여행지 추천 시 LLM(타임아웃 적용) 또는 키워드 fallback으로 Who/Why/Constraints/When/대화단계를 분류합니다."""
    user_message = state.get("latest_message", "")
    llm = _get_travel_llm()
    timeout_sec = _llm_timeout_sec()

    if llm is None:
        data = _fallback_classify_travel(user_message)
        state["who"] = data.get("who") or "unknown"
        state["why"] = data.get("why") or "unknown"
        state["constraints"] = data.get("constraints") or {}
        state["when_info"] = data.get("when_info") or {}
        state["conversation_stage"] = data.get("conversation_stage") or "exploration"
        return state

    history = state.get("user_input", [])
    recent = history[-4:] if len(history) > 4 else history
    context = "\n".join(
        f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}"
        for m in recent
    ) if recent else "(대화 없음)"
    prompt = f"""대화 맥락:\n{context}\n\n현재 사용자 발화: {user_message}\n\n위 규칙에 따라 JSON만 출력하세요."""

    try:
        msgs = [SystemMessage(content=TRAVEL_CLASSIFIER_SYSTEM), HumanMessage(content=prompt)]
        response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout_sec)
        text = response.content if hasattr(response, "content") else str(response)
        raw = text.strip()
        if "```" in raw:
            start = raw.find("```") + 3
            if raw[start:start+4].lower() == "json":
                start += 4
            end = raw.find("```", start)
            raw = raw[start:end if end > 0 else None].strip()
        data = json.loads(raw)
        state["who"] = data.get("who") or "unknown"
        state["why"] = data.get("why") or "unknown"
        state["constraints"] = data.get("constraints") or {}
        state["when_info"] = data.get("when_info") or {}
        state["conversation_stage"] = data.get("conversation_stage") or "exploration"
    except asyncio.TimeoutError:
        print(f"[graph] LLM 분류 타임아웃 ({timeout_sec}초 초과) → fallback 적용")
        data = _fallback_classify_travel(user_message)
        state["who"] = data.get("who") or "unknown"
        state["why"] = data.get("why") or "unknown"
        state["constraints"] = data.get("constraints") or {}
        state["when_info"] = data.get("when_info") or {}
        state["conversation_stage"] = data.get("conversation_stage") or "exploration"
    except Exception as e:
        print(f"[graph] LLM 분류 에러: {type(e).__name__} {e} → fallback 적용")
        data = _fallback_classify_travel(user_message)
        state["who"] = data.get("who") or "unknown"
        state["why"] = data.get("why") or "unknown"
        state["constraints"] = data.get("constraints") or {}
        state["when_info"] = data.get("when_info") or {}
        state["conversation_stage"] = data.get("conversation_stage") or "exploration"
    return state


# ==================== 1. Entry / Input 노드 ====================
def user_input_node(state: TravelState) -> TravelState:
    """
    사용자 입력을 수집하고 상태에 병합합니다.
    """
    # 최신 메시지가 있으면 히스토리에 추가
    if state.get("latest_message"):
        message = HumanMessage(content=state["latest_message"])
        current_messages = state.get("user_input", [])
        state["user_input"] = current_messages + [message]
    
    return state


# ==================== 2. Intent 분류 노드 ====================
def intent_classifier_node(state: TravelState) -> TravelState:
    """
    사용자 입력을 분석하여 의도를 분류합니다.
    """
    user_message = state.get("latest_message", "").lower()
    
    # 간단한 키워드 기반 의도 분류 (실제로는 LLM 사용 권장)
    intent_keywords = {
        "recommend_place": ["여행지", "관광지", "가볼만한", "추천", "어디"],
        "recommend_accommodation": ["숙소", "호텔", "펜션", "게스트하우스", "예약", "잠자리"],
        "add_favorite": ["찜", "저장", "즐겨찾기", "북마크"],
        "show_favorites": ["찜 목록", "저장한", "즐겨찾기 목록", "찜 보기"],
        "plan_trip": ["일정", "계획", "여행 계획", "스케줄"]
    }
    
    # 키워드 매칭으로 의도 분류
    detected_intent = None
    for intent, keywords in intent_keywords.items():
        if any(keyword in user_message for keyword in keywords):
            detected_intent = intent
            break
    
    # 기본값: 여행지 추천
    if not detected_intent:
        detected_intent = "recommend_place"
    
    state["intent"] = detected_intent
    return state


# ==================== 3. 정보 추출 노드 (NER / 슬롯 추출 + LLM 결과 병합) ====================
def travel_info_extractor_node(state: TravelState) -> TravelState:
    """
    사용자 입력에서 지역/테마 등 규칙 추출 후, LLM 분류 결과(who/why/constraints/when)를 filters에 병합합니다.
    """
    user_message = state.get("latest_message", "")
    filters = dict(state.get("filters", {}))
    
    # 지역 추출
    regions = ["부산", "제주", "서울", "경주", "강릉", "여수", "전주", "대구", "인천"]
    for region in regions:
        if region in user_message:
            filters["region"] = region
            break
    
    # 테마 추출
    themes = []
    theme_keywords = {
        "바다": ["바다", "해변", "해수욕장", "해안"],
        "감성": ["감성", "로맨틱", "분위기"],
        "힐링": ["힐링", "휴식", "여유"],
        "커플": ["커플", "연인", "데이트"],
        "가족": ["가족", "아이", "어린이"],
        "맛집": ["맛집", "음식", "식당"],
        "야경": ["야경", "밤", "야경"]
    }
    for theme, keywords in theme_keywords.items():
        if any(keyword in user_message for keyword in keywords):
            themes.append(theme)
    if themes:
        filters["theme"] = themes
    
    # 인원/예산 규칙 추출
    if "1명" in user_message or "혼자" in user_message:
        filters["people"] = "1명"
    elif "2명" in user_message or "커플" in user_message or "둘이" in user_message:
        filters["people"] = "2명"
    elif "가족" in user_message or "아이" in user_message:
        filters["people"] = "가족"
    if "저렴" in user_message or "싼" in user_message or "경제적" in user_message:
        filters["budget"] = "저렴"
    elif "비싼" in user_message or "고급" in user_message or "럭셔리" in user_message:
        filters["budget"] = "고급"
    
    # LLM 분류 결과 병합: who, why, constraints, when_info → filters
    who = state.get("who")
    if who and who != "unknown":
        filters["who"] = who
    why = state.get("why")
    if why and why != "unknown":
        filters["why"] = why
    for k, v in (state.get("constraints") or {}).items():
        if v is not None:
            filters[f"constraint_{k}"] = v
    when_info = state.get("when_info") or {}
    if when_info.get("season"):
        filters["season"] = when_info["season"]
    if when_info.get("need_night_info") is True:
        filters["need_night_info"] = True
    cs = state.get("conversation_stage")
    if cs:
        filters["conversation_stage"] = cs
    
    state["filters"] = filters
    return state


# ==================== 4. 정보 부족 여부 판단 노드 ====================
def missing_info_check_node(state: TravelState) -> TravelState:
    """
    필수 정보(지역·테마·Who·Why·제약·시기)가 충분한지 확인하고, 부족한 항목을 식별합니다.
    """
    filters = state.get("filters", {})
    missing_info = []
    intent = state.get("intent", "")
    
    if "recommend_place" in intent or "recommend_accommodation" in intent or "plan_trip" in intent:
        # 필수: 지역, 테마가 있으면 RAG 진행 가능
        if "region" not in filters:
            missing_info.append("region")
        if "theme" not in filters:
            missing_info.append("theme")
        # Who/Why: unknown이면 추천 정교화를 위해 질문 후보로 추가
        if state.get("who") in (None, "unknown"):
            missing_info.append("who")
        if state.get("why") in (None, "unknown"):
            missing_info.append("why")
        # 제약·시기: LLM이 못 채웠을 때만 추가 (이동수단·예산·계절)
        constraints = state.get("constraints") or {}
        if not constraints.get("transport"):
            missing_info.append("transport")
        if not constraints.get("budget") and not filters.get("budget"):
            missing_info.append("budget")
        when_info = state.get("when_info") or {}
        if not when_info.get("season"):
            missing_info.append("season")
    
    state["missing_info"] = missing_info
    return state


# ==================== 5. 추가 질문 생성 노드 (여행 전문가 대화 톤) ====================
TRAVEL_CONSULTANT_RESPONSE = """당신은 친절한 여행 전문가입니다. 전화로 대화하듯이 **한 문단의 자연스러운 말**만 하세요.

규칙:
- JSON, 불릿, 번호 목록, " | " 로 질문 나열 금지.
- 공감 한마디 후, 부족한 정보를 **한두 가지**만 부드럽게 여쭤보세요.
- 예: "그러셨군요! 그런데 어디쯤 가고 싶으세요? 부산·제주처럼 바다가 좋을까요, 아니면 서울 근교가 더 나을까요?"
- 다른 설명·메타 문구 없이, 사용자에게 할 **응답 문장만** 출력하세요."""


def _fallback_conversational_question(missing_info: list, user_message: str, filters: dict, who: str, why: str) -> str:
    """LLM 없을 때 여행 전문가처럼 한 문장~한 문단으로 질문을 만듭니다. 'Q1 | Q2' 형태 금지."""
    if not missing_info:
        return ""
    # 우선순위: region > theme > who > why > 나머지
    order = ["region", "theme", "who", "why", "season", "transport", "budget", "people", "duration", "pet_friendly"]
    todo = [m for m in order if m in missing_info][:2]
    if not todo:
        return ""

    msg = (user_message or "").strip()[:80]
    if "region" in todo:
        if "theme" in todo:
            return "어디로 가고 싶으세요? 부산, 제주, 서울 같은 데 중에 생각 중인 곳이 있으시면 말해 주세요. 그리고 바다 보면서 힐링할까요, 아니면 맛집·감성 분위기 쪽이 더 끌리세요?"
        return "어느 지역이 좋을까요? 부산, 제주, 강릉처럼 바다가 보이는 쪽이요, 아니면 서울·경기 쪽이 더 나을까요?"
    if "theme" in todo:
        return "어떤 분위기가 더 끌리세요? 바다 보며 힐링, 맛집·카페 투어, 아니면 액티비티나 문화 체험 쪽이요?"
    if "who" in todo:
        return "누구와 함께 가시나요? 커플이나 가족, 혼자 여행이요? 그거에 따라 어디가 좋을지가 많이 달라져서요."
    if "why" in todo:
        return "이번엔 뭐가 가장 중요하세요? 휴식·힐링, 맛집·미식, 아니면 볼거리·체험 같은 거요?"
    if "season" in todo:
        return "어느 시즌에 가실 예정이에요? 봄·여름·가을·겨울에 따라 추천할 코스가 조금씩 달라져요."
    if "transport" in todo:
        return "이동은 차로 하실까요, 아니면 대중교통 위주로 생각하고 계세요?"
    if "budget" in todo:
        return "예산은 어느 정도로 생각하고 계세요? 가성비 위주로 잡을까요, 아니면 좀 여유 있게 가실 계획이세요?"

    return "조금만 더 알려주시면 딱 맞는 곳 골라 드릴게요. 어디로, 어떤 분위기로 가고 싶으세요?"


async def clarifying_question_node(state: TravelState) -> TravelState:
    """
    부족한 정보를 여행 전문가처럼 한 문단의 자연스러운 말로 물어봅니다.
    JSON·불릿·'|' 나열 없이, 대화체 한 문단만 출력합니다.
    """
    missing_info = state.get("missing_info", [])
    filters = state.get("filters", {})
    user_message = state.get("latest_message", "")
    who = state.get("who") or "unknown"
    why = state.get("why") or "unknown"

    if not missing_info:
        state["clarifying_question"] = None
        state["response"] = None
        return state

    # 확보된 정보 / 부족한 정보 정리
    confirmed = []
    if filters.get("region"):
        confirmed.append(f"지역-{filters['region']}")
    if filters.get("theme"):
        confirmed.append(f"테마-{','.join(filters['theme'])}")
    if who and who != "unknown":
        confirmed.append(f"동행-{who}")
    if why and why != "unknown":
        confirmed.append(f"목적-{why}")
    missing_labels = [m for m in missing_info if m in ("region", "theme", "who", "why", "season", "transport", "budget", "people", "duration", "pet_friendly")]

    llm = _get_travel_llm()
    timeout_sec = _llm_timeout_sec()

    if llm and missing_labels:
        try:
            user_prompt = f"""현재 확보된 정보: {", ".join(confirmed) if confirmed else "아직 없음"}
부족한 정보: {", ".join(missing_labels)}
사용자 방금 한 말: {user_message[:150]}

위 규칙대로, **한 문단의 자연스러운 응답만** 작성하세요. 다른 설명이나 JSON·목록 금지."""
            msgs = [SystemMessage(content=TRAVEL_CONSULTANT_RESPONSE), HumanMessage(content=user_prompt)]
            response = await asyncio.wait_for(llm.ainvoke(msgs), timeout=timeout_sec)
            text = (response.content if hasattr(response, "content") else str(response)).strip()
            if text and len(text) > 10:
                state["clarifying_question"] = text
                state["response"] = text
                return state
        except (asyncio.TimeoutError, Exception):
            pass

    # fallback: 대화체 한 문단
    out = _fallback_conversational_question(missing_info, user_message, filters, who, why)
    if out:
        state["clarifying_question"] = out
        state["response"] = out
    else:
        state["clarifying_question"] = None
    return state


# ==================== 6. RAG 검색 노드 (Who/Why/Constraints/When 반영 Mock) ====================
# who: family_with_kids | couple | parents_trip | solo
# why: relaxation | activity | culture | food
# transport_ok: ["car"] | ["public"] | ["car","public"]
# budget: "value" | "luxury" | "both"
# pet_friendly: True | False
# season_ok: ["spring","summer","autumn","winter"] 중 해당 계절
MOCK_DOCS = [
    # 부산·바다·감성·커플/나홀
    {
        "name": "해운대 달맞이길",
        "region": "부산",
        "theme": ["바다", "감성"],
        "description": "부산 해운대의 아름다운 해안 산책로. 야경과 로맨틱한 분위기.",
        "who": ["couple", "solo"],
        "why": ["relaxation"],
        "transport_ok": ["car", "public"],
        "budget": "both",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn"],
        "score": 0.95,
    },
    {
        "name": "감천문화마을",
        "region": "부산",
        "theme": ["감성", "힐링"],
        "description": "부산의 대표 감성 관광지. 골목길 산책, 포토존, 로컬 맛집.",
        "who": ["couple", "solo", "parents_trip"],
        "why": ["relaxation", "culture", "food"],
        "transport_ok": ["public"],
        "budget": "value",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn", "winter"],
        "score": 0.92,
    },
    {
        "name": "송도 스카이워크",
        "region": "부산",
        "theme": ["바다", "야경"],
        "description": "바다 위를 걷는 스카이워크. 밤에도 개장하여 야경 코스로 인기.",
        "who": ["couple", "solo"],
        "why": ["activity", "relaxation"],
        "transport_ok": ["car", "public"],
        "budget": "value",
        "pet_friendly": False,
        "season_ok": ["summer", "autumn"],
        "score": 0.88,
    },
    # 부산·가족·액티비티
    {
        "name": "부산 아쿠아리움",
        "region": "부산",
        "theme": ["바다", "가족"],
        "description": "실내 수족관. 유모차 이동 편하고 키즈존·체험형 프로그램 보유.",
        "who": ["family_with_kids"],
        "why": ["activity", "relaxation"],
        "transport_ok": ["car", "public"],
        "budget": "both",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn", "winter"],
        "score": 0.90,
    },
    {
        "name": "기장 스카이라인 루지",
        "region": "부산",
        "theme": ["액티비티", "가족"],
        "description": "레일 루지·짚라인. 가족 단위 체험 코스와 주차 편의.",
        "who": ["family_with_kids", "couple"],
        "why": ["activity"],
        "transport_ok": ["car"],
        "budget": "value",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn"],
        "score": 0.87,
    },
    # 부산·효도·휴양
    {
        "name": "해운대 온천센터",
        "region": "부산",
        "theme": ["힐링", "휴양"],
        "description": "걷기 편한 입장로, 탕류 다양. 부모님 효도 코스로 추천.",
        "who": ["parents_trip", "couple"],
        "why": ["relaxation"],
        "transport_ok": ["car", "public"],
        "budget": "both",
        "pet_friendly": False,
        "season_ok": ["spring", "autumn", "winter"],
        "score": 0.91,
    },
    {
        "name": "영도 한밭한식당",
        "region": "부산",
        "theme": ["맛집", "한식"],
        "description": "로컬 한식 맛집. 조용한 분위기로 부모님과 방문하기 좋음.",
        "who": ["parents_trip", "solo"],
        "why": ["food", "relaxation"],
        "transport_ok": ["car", "public"],
        "budget": "value",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn", "winter"],
        "score": 0.86,
    },
    # 제주·나홀·힐링/미식
    {
        "name": "제주 협재 해변 카페거리",
        "region": "제주",
        "theme": ["바다", "힐링", "맛집"],
        "description": "혼밥·혼카페하기 좋은 해변가. 치안 좋고 게스트하우스 인접.",
        "who": ["solo", "couple"],
        "why": ["relaxation", "food"],
        "transport_ok": ["car"],
        "budget": "value",
        "pet_friendly": True,
        "season_ok": ["summer", "autumn"],
        "score": 0.89,
    },
    {
        "name": "제주 동문시장",
        "region": "제주",
        "theme": ["맛집", "문화"],
        "description": "전통시장·로컬 노포. 미식·문화 테마에 맞음.",
        "who": ["solo", "couple", "parents_trip"],
        "why": ["food", "culture"],
        "transport_ok": ["car", "public"],
        "budget": "value",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn", "winter"],
        "score": 0.88,
    },
    # 강릉·커플/힐링
    {
        "name": "강릉 경포대 카페거리",
        "region": "강릉",
        "theme": ["감성", "바다", "힐링"],
        "description": "바다 뷰 카페와 조용한 산책로. 커플·나홀 힐링에 적합.",
        "who": ["couple", "solo"],
        "why": ["relaxation", "food"],
        "transport_ok": ["car", "public"],
        "budget": "both",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn"],
        "score": 0.90,
    },
    {
        "name": "강릉 정동진 해돋이",
        "region": "강릉",
        "theme": ["바다", "감성"],
        "description": "해돋이 명소. 새벽 개장으로 야간·시간대 정보 중요.",
        "who": ["couple", "solo", "parents_trip"],
        "why": ["relaxation", "culture"],
        "transport_ok": ["car"],
        "budget": "value",
        "pet_friendly": False,
        "season_ok": ["spring", "summer", "autumn", "winter"],
        "score": 0.87,
    },
]


def retrieval_node(state: TravelState) -> TravelState:
    """
    벡터 DB에서 관련 문서를 검색합니다. Who/Why/Constraints/When 조건에 맞는 mock_docs만 반환합니다.
    """
    filters = state.get("filters", {})
    region = filters.get("region", "")
    themes = filters.get("theme", [])
    who = state.get("who") or filters.get("who")
    why = state.get("why") or filters.get("why")
    constraint_transport = (state.get("constraints") or {}).get("transport") or filters.get("constraint_transport")
    constraint_budget = (state.get("constraints") or {}).get("budget") or filters.get("budget") or filters.get("constraint_budget")
    pet_friendly = (state.get("constraints") or {}).get("pet_friendly")
    when_info = state.get("when_info") or {}
    season = when_info.get("season") or filters.get("season")

    filtered_docs = []
    for doc in MOCK_DOCS:
        doc_who = doc.get("who") or []
        doc_why = doc.get("why") or []
        doc_transport = doc.get("transport_ok") or []
        doc_budget = doc.get("budget") or "both"
        doc_pet = doc.get("pet_friendly")
        doc_season = doc.get("season_ok") or []

        if region and doc.get("region") != region:
            continue
        if themes and not any(t in doc.get("theme", []) for t in themes):
            continue
        if who and who != "unknown" and who not in doc_who:
            continue
        if why and why != "unknown" and why not in doc_why:
            continue
        if constraint_transport and constraint_transport not in doc_transport:
            continue
        if constraint_budget and doc_budget != "both" and doc_budget != constraint_budget:
            continue
        if pet_friendly is True and doc_pet is not True:
            continue
        if pet_friendly is False and doc_pet is True:
            continue
        if season and season not in doc_season:
            continue
        filtered_docs.append(doc)

    # 점수순 정렬 후 상위 5개
    filtered_docs.sort(key=lambda d: d.get("score", 0), reverse=True)
    state["retrieved_docs"] = filtered_docs[:5]
    return state


# ==================== 7. 추천 생성 노드 (LLM) ====================
def recommendation_node(state: TravelState) -> TravelState:
    """
    RAG 결과와 사용자 조건(who/why/constraints/when 포함)을 바탕으로 최종 추천을 생성합니다.
    """
    retrieved_docs = state.get("retrieved_docs", [])
    filters = state.get("filters", {})
    who = state.get("who") or filters.get("who")
    why = state.get("why") or filters.get("why")
    
    who_labels = {"family_with_kids": "아이 동반 가족", "couple": "커플·신혼", "parents_trip": "효도 여행", "solo": "나홀", "unknown": ""}
    why_labels = {"relaxation": "휴식·힐링", "activity": "액티비티", "culture": "문화·역사", "food": "미식", "unknown": ""}
    
    recommendations = []
    for doc in retrieved_docs[:3]:  # Top 3 추천
        reason_parts = []
        if filters.get("region"):
            reason_parts.append(f"{filters['region']} 지역")
        if filters.get("theme"):
            reason_parts.append(f"{', '.join(filters['theme'])} 테마")
        if who and who != "unknown" and who_labels.get(who):
            reason_parts.append(who_labels[who])
        if why and why != "unknown" and why_labels.get(why):
            reason_parts.append(why_labels[why])
        reason = " + ".join(reason_parts) if reason_parts else "사용자 조건에 맞는 추천"
        
        recommendations.append({
            "name": doc.get("name", ""),
            "region": doc.get("region", ""),
            "theme": doc.get("theme", []),
            "description": doc.get("description", ""),
            "reason": reason,
            "score": doc.get("score", 0.0),
            "who": doc.get("who", []),
            "why": doc.get("why", []),
        })
    
    state["recommendations"] = recommendations
    
    # 응답 메시지 생성
    if recommendations:
        response_parts = [f"추천 드리는 {len(recommendations)}곳입니다:\n\n"]
        for i, rec in enumerate(recommendations, 1):
            response_parts.append(
                f"{i}. {rec['name']}\n"
                f"   - {rec['description']}\n"
                f"   - 추천 이유: {rec['reason']}\n"
            )
        state["response"] = "\n".join(response_parts)
    else:
        state["response"] = "조건에 맞는 추천을 찾지 못했습니다. 다른 조건으로 검색해보시겠어요?"
    
    return state


# ==================== 8. 후속 액션 유도 노드 ====================
def post_action_node(state: TravelState) -> TravelState:
    """
    사용자에게 후속 액션을 제안합니다.
    """
    post_actions = [
        "지도에서 위치 확인하기",
        "찜 목록에 추가하기",
        "비슷한 스타일 더 추천받기",
        "상세 정보 보기"
    ]
    
    state["post_actions"] = post_actions
    
    # 응답에 후속 액션 추가
    if state.get("response"):
        action_text = "\n\n추가로 도와드릴까요?\n" + "\n".join([f"- {action}" for action in post_actions])
        state["response"] += action_text
    
    return state


# ==================== 9. 찜 추가 노드 ====================
def add_favorite_node(state: TravelState) -> TravelState:
    """
    사용자가 선택한 항목을 찜 목록에 추가합니다.
    """
    user_id = state.get("user_id")
    user_message = state.get("latest_message", "")
    
    if not user_id:
        state["response"] = "찜 기능을 사용하려면 로그인이 필요합니다."
        return state
    
    # TODO: 실제 DB 저장 로직
    # favorite_service.add_favorite(user_id, item_id)
    
    # 메시지에서 항목 이름 추출 시도
    item_name = None
    for rec in state.get("recommendations", []):
        if rec["name"] in user_message:
            item_name = rec["name"]
            break
    
    if item_name:
        state["response"] = f"'{item_name}'을(를) 찜 목록에 추가했습니다!"
    else:
        state["response"] = "어떤 항목을 찜 목록에 추가하시겠어요?"
    
    return state


# ==================== 10. 찜 목록 조회 노드 ====================
def favorite_list_node(state: TravelState) -> TravelState:
    """
    사용자의 찜 목록을 조회합니다.
    """
    user_id = state.get("user_id")
    
    if not user_id:
        state["response"] = "찜 목록을 보려면 로그인이 필요합니다."
        return state
    
    # TODO: 실제 DB 조회 로직
    # favorites = favorite_service.get_favorites(user_id)
    
    # Mock 데이터
    mock_favorites = [
        {"name": "해운대 달맞이길", "added_at": "2025-01-20"},
        {"name": "감천문화마을", "added_at": "2025-01-18"}
    ]
    
    state["favorite_items"] = mock_favorites
    
    if mock_favorites:
        response_parts = ["찜 목록입니다:\n"]
        for i, fav in enumerate(mock_favorites, 1):
            response_parts.append(f"{i}. {fav['name']} (추가일: {fav['added_at']})")
        state["response"] = "\n".join(response_parts)
    else:
        state["response"] = "찜 목록이 비어있습니다."
    
    return state
