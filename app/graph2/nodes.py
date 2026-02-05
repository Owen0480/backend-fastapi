import json
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings
from app.graph2.state import AgentState

llm = ChatGroq(
    model_name="llama-3.1-8b-instant", 
    groq_api_key=settings.GROQ_API_KEY,
    temperature=0.4
)

# ============== 유틸리티 함수 ==============

def extract_json(text: str):
    """문자열에서 JSON 블록을 추출하여 파싱합니다."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # JSON 전후에 텍스트가 섞여 있을 경우 { } 를 찾아 시도
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        # 배열 형태 [ ] 도 시도
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
        raise

# ============== 노드 함수들 ==============

async def intent_classifier_node(state: AgentState) -> AgentState:
    """사용자의 입력을 분류하는 노드"""
    print("\n=== 의도 분류 중 ===")
    
    messages = state.get("messages", [])
    if not messages:
        return AgentState(intent="general_chat", current_step="intent_classifier")
    
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        user_input = last_msg.get("content", "")
    else:
        user_input = getattr(last_msg, "content", str(last_msg))
        
    system_prompt = """당신은 사용자의 입력 의도를 분류하는 분류기입니다.
    다음 세 가지 카테고리 중 하나로 분류하세요:
    1. "recommend_travel": 여행지와 관련된 모든 질문, 여행 추천 요청, 여행 계획 관련 대화 등
    2. "general_chat": 인사(안녕하세요, 반가워요), 시스템 관련 질문(넌 누구니?), 칭찬, 간단한 일상 대화 등
    3. "irrelevant_chat": 여행과 전혀 상관없는 주제(정치, 기술 질문, 요리 레시피 등)나 부적절한 언어

    응답은 오직 한 단어(recommend_travel, general_chat, irrelevant_chat)로만 하세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    intent = response.content.strip().lower()
    # 엄격한 필터링: 예상치 못한 응답 대비
    if "recommend" in intent:
        intent = "recommend_travel"
    elif "general" in intent:
        intent = "general_chat"
    elif "irrelevant" in intent:
        intent = "irrelevant_chat"
    else:
        intent = "recommend_travel" # 기본값
        
    print(f"분류된 의도: {intent}")
    return AgentState(intent=intent, current_step="intent_classifier")


async def general_chat_node(state: AgentState) -> AgentState:
    """간단한 인사나 일상 대화에 응답하는 노드"""
    print("\n=== 일상 대화 응답 중 ===")
    
    messages = state.get("messages", [])
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        user_input = last_msg.get("content", "")
    else:
        user_input = getattr(last_msg, "content", str(last_msg))
        
    system_prompt = """당신은 친절한 여행 전문가 어시스턴트입니다. 
    사용자의 인사나 가벼운 질문에 친절하고 짧게 대답하세요. 
    그리고 자연스럽게 국내 여행지 추천이 필요하면 언제든 말씀해달라고 덧붙이세요.
    이모지를 적절히 사용하여 따뜻한 분위기를 만드세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
    
    return AgentState(
        messages=[response.content],
        current_step="general_chat"
    )


async def irrelevant_chat_node(state: AgentState) -> AgentState:
    """주제에서 벗어난 질문에 대해 가이드를 주는 노드"""
    print("\n=== 부적합 대화 안내 중 ===")
    
    system_prompt = """당신은 '대한민국 국내 여행 전문가'입니다. 
    사용자가 여행과 관련 없는 질문을 하거나 부적절한 말을 했을 때, 
    정중하게 당신의 역할을 설명하고 여행에 관한 질문만 해달라고 안내하세요.
    단호하지만 친절한 말투를 유지하세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="여행과 상관없는 질문을 하거나 부적절한 대화를 시도함.")
    ])
    
    return AgentState(
        messages=[response.content],
        current_step="irrelevant_chat"
    )


async def collect_preferences_node(state: AgentState) -> AgentState:
    """사용자 선호도 수집 노드"""
    print("\n=== 입력 수집 중 ===")
    
    system_prompt = """당신은 대한민국 국내 여행 전문가입니다. 사용자의 여행 선호도를 파악해주세요.
    다음 정보가 필요합니다:
    - budget (예산)
    - duration (여행 기간, 일 단위)
    - interests (관심사: 문화, 자연, 음식, 액티비티 등)
    - travel_style (여행 스타일: 휴양, 모험, 관광 등)
    - season (선호 계절 또는 여행 시기)
    - companion (동행인: 혼자, 가족, 연인, 친구 등)
    
    사용자 메시지를 분석해서 JSON 형태로 추출하세요.
    부족한 정보가 있으면 필요한 질문도 함께 제시하세요."""
    
    # 필수 정보가 완벽하지 않은 경우 계속해서 수집/업데이트 시도
    required_fields = ["budget", "interests"]
    current_prefs = state.get("user_preferences", {})
    has_all = all(field in current_prefs and current_prefs[field] for field in required_fields)
    
    if not has_all:
        messages = state.get("messages", [])
        if not messages:
            return AgentState(
                user_preferences={},
                messages=["안녕하세요! 완벽한 여행지를 추천해드리겠습니다. 예산, 여행 기간, 관심사를 알려주세요."],
                current_step="collect_preferences"
            )
        
        # 유저의 마지막 답변 가져오기
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            user_input = last_msg.get("content", "")
        else:
            user_input = getattr(last_msg, "content", str(last_msg))
            
        # 기존에 알고 있는 정보도 함께 전달하여 문맥 유지
        try:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"기존 추출 정보: {json.dumps(current_prefs, ensure_ascii=False)}\n사용자 입력: {user_input}\n\n새로운 정보를 반영하여 JSON으로 출력하고, 여전히 부족한 정보는 'missing_fields'에 나열하세요. 다른 설명 없이 오직 JSON만 반환하세요.")
            ])
        except Exception as e:
            print(f"LLM 호출 실패: {e}")
            return AgentState(
                messages=["정보를 처리하는 중 오류가 발생했습니다."],
                current_step="collect_preferences"
            )

        # 응답 파싱
        try:
            result = extract_json(response.content)
            # 기존 정보와 새로 추출된 정보 병합
            new_preferences = {**current_prefs}
            for k, v in result.items():
                if k != "missing_fields" and v:
                    new_preferences[k] = v
                    
            missing = result.get("missing_fields", [])
            
            # 병합된 정보로 다시 한번 누락 확인
            final_missing = [f for f in required_fields if f not in new_preferences or not new_preferences[f]]
            
            if final_missing:
                return AgentState(
                    user_preferences=new_preferences,
                    messages=[f"좋습니다! 추가로 다음 정보가 필요합니다: {', '.join(final_missing)}"],
                    current_step="collect_preferences"
                )
            
            # 필수 정보 충족도 계산
            quality = sum(1 for f in required_fields if f in new_preferences) / len(required_fields)
            
            return AgentState(
                user_preferences=new_preferences,
                info_quality_score=quality,
                messages=["모든 정보 수집 완료! 맞춤 여행지를 찾고 있습니다..."],
                current_step="collect_preferences"
            )
        except Exception as e:
            print(f"파싱 오류: {e}")
            return AgentState(
                messages=["정보를 이해하지 못했습니다. 예산, 관심사(문화, 자연, 음식 등)를 다시 말씀해주세요."],
                current_step="collect_preferences"
            )
    
    return AgentState(current_step="collect_preferences")


async def generate_candidates_node(state: AgentState) -> AgentState:
    """여행지 후보 생성 노드"""
    print("\n=== 후보지 생성 중 ===")
    
    prefs = state["user_preferences"]
    retry_count = state.get("retry_count", 0)
    
    system_prompt = f"""당신은 대한민국 구석구석을 꿰뚫고 있는 국내 여행 전문가입니다. 
    사용자의 선호도를 기반으로 최적의 **국내 여행지(한국 내 도시 및 지역)** 3-5개를 추천하세요.

    사용자 정보:
    {json.dumps(prefs, ensure_ascii=False, indent=2)}

    {"[주의] 이전 추천이 선호도와 맞지 않았습니다. 이번에는 이전에 언급하지 않았던 새로운 지역이나 다른 테마의 한국 여행지를 추천해주세요." if retry_count > 0 else ""}

    각 여행지에 대해 다음을 포함하세요:
    - destination: 여행지 이름 (예: 제주도, 경주, 양양 등)
    - country: "대한민국"으로 고정
    - province: 도 단위 (예: 강원도, 전라남도, 제주특별자치도 등)
    - reason: 추천 이유 (사용자의 관심사와 해당 지역의 특성을 구체적으로 연결)
    - estimated_cost: 예상 비용 범위 (단위: 원)
    - best_season: 최적 방문 시기
    - highlights: 주요 볼거리/명소 3가지

    응답은 반드시 JSON 배열 형태([ ... ])로만 하세요. 다른 설명이나 텍스트는 절대 포함하지 마세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="여행지를 추천해주세요.")
    ])
    
    try:
        candidates = extract_json(response.content)
        print(f"생성된 후보: {len(candidates)}개")
        
        return AgentState(
            candidates=candidates,
            messages=[f"{len(candidates)}개의 여행지 후보를 생성했습니다."],
            current_step="generate_candidates"
        )
    except Exception as e:
        print(f"후보 생성 오류: {e}")
        return AgentState(
            candidates=[],
            validation_score=0.0,
            validation_feedback="후보 생성 실패",
            messages=["후보 생성에 실패했습니다."],
            current_step="generate_candidates"
        )


async def validate_candidates_node(state: AgentState) -> AgentState:
    """후보지 품질 검증 노드"""
    print("\n=== 후보 검증 중 ===")
    
    candidates = state["candidates"]
    prefs = state["user_preferences"]
    
    if not candidates:
        return AgentState(
            validation_score=0.0,
            validation_feedback="후보가 없습니다",
            messages=["검증할 후보가 없습니다."],
            current_step="validate_candidates"
        )
    
    system_prompt = f"""후보 여행지의 품질을 평가하세요.
    
    사용자 선호도:
    {json.dumps(prefs, ensure_ascii=False, indent=2)}
    
    후보 목록:
    {json.dumps(candidates, ensure_ascii=False, indent=2)}
    
    다음 기준으로 평가하세요:
    1. 사용자 요구사항 일치도 (예산, 관심사, 기간) - 40점
    2. 후보 다양성 (지역, 스타일의 다양성) - 30점
    3. 실현 가능성 (비자, 접근성, 안전) - 20점
    4. 정보 구체성 (상세하고 유용한 정보) - 10점
    
    JSON으로만 응답:
    {{
        "score": 0.0-1.0 (소수점),
        "feedback": "평가 설명",
        "issues": ["문제점 리스트"]
    }}"""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="후보를 평가해주세요.")
    ])
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        validation = json.loads(content.strip())
        score = float(validation.get("score", 0.0))
        feedback = validation.get("feedback", "")
        
        print(f"검증 점수: {score:.2f}")
        print(f"피드백: {feedback}")
        
        return AgentState(
            validation_score=score,
            validation_feedback=feedback,
            messages=[f"후보 검증 완료 (점수: {score:.2f})"],
            current_step="validate_candidates"
        )
    except Exception as e:
        print(f"검증 오류: {e}")
        return AgentState(
            validation_score=0.5,
            validation_feedback="검증 중 오류 발생",
            messages=["검증에 실패했습니다."],
            current_step="validate_candidates"
        )

async def enrich_information_node(state: AgentState) -> AgentState:
    """여행지 정보 보강 노드"""
    print("\n=== 정보 수집 중 ===")
    
    candidates = state["candidates"]
    
    # 실제로는 web_search 도구를 사용할 수 있습니다
    system_prompt = """각 여행지에 대한 추가 정보를 제공하세요:
    - weather: 현재 계절의 날씨/기후 정보
    - safety: 안전 정보 및 주의사항
    - transport: 교통 정보 (공항, 대중교통)
    - tips: 여행 팁 (현지 문화, 추천 음식 등)
    - recent_reviews: 최근 여행자 피드백 요약
    
    각 destination에 대해 enriched_info 필드를 추가한 JSON 배열로 응답하세요.
    원본 정보는 유지하고 enriched_info만 추가하세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"다음 여행지들의 정보를 보강하세요:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}")
    ])
    
    try:
        enriched = extract_json(response.content)
        print(f"정보 보강 완료: {len(enriched)}개")
        
        return AgentState(
            enriched_data=enriched,
            messages=["여행지 정보를 업데이트했습니다."],
            current_step="enrich_information"
        )
    except Exception as e:
        print(f"정보 보강 오류: {e}")
        return AgentState(
            enriched_data=candidates,
            messages=["정보 보강을 건너뛰었습니다."],
            current_step="enrich_information"
        )


async def validate_information_node(state: AgentState) -> AgentState:
    """수집된 정보의 품질 검증 노드"""
    print("\n=== 정보 품질 검증 중 ===")
    
    enriched = state["enriched_data"]
    
    system_prompt = """수집된 여행지 정보의 품질을 평가하세요.
    
    평가 기준:
    - 정보의 구체성 (모호하지 않고 구체적인가)
    - 최신성 (최근 정보인가)
    - 완성도 (필요한 정보가 모두 있는가)
    
    JSON으로만 응답:
    {
        "quality_score": 0.0-1.0,
        "assessment": "평가 내용"
    }"""
    
    # 전체가 너무 크면 샘플만 검증
    sample = enriched[:3] if len(enriched) > 3 else enriched
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"정보를 평가하세요:\n{json.dumps(sample, ensure_ascii=False, indent=2)}")
    ])
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        score = float(result.get("quality_score", 0.8))
        
        print(f"정보 품질 점수: {score:.2f}")
        
        return AgentState(
            info_quality_score=score,
            messages=[f"정보 품질 검증 완료 (점수: {score:.2f})"],
            current_step="validate_information"
        )
    except Exception as e:
        print(f"정보 검증 오류: {e}")
        return AgentState(
            info_quality_score=0.7,
            messages=["정보 검증을 건너뛰었습니다."],
            current_step="validate_information"
        )

async def filter_options_node(state: AgentState) -> AgentState:
    """Hard constraint로 필터링하는 노드"""
    print("\n=== 옵션 필터링 중 ===")
    
    enriched = state["enriched_data"]
    prefs = state["user_preferences"]
    
    system_prompt = f"""사용자의 필수 조건에 맞지 않는 여행지를 제거하세요.
    
    사용자 조건:
    {json.dumps(prefs, ensure_ascii=False, indent=2)}
    
    필터링 기준:
    - 예산 초과 (예산의 120%를 넘으면 제거)
    - 계절/시기 부적합 (완전히 맞지 않는 경우만)
    - 안전 문제 (심각한 경우만)
    - 기간 부적합 (너무 짧거나 길면)
    
    적합한 여행지만 JSON 배열로 반환하세요. 다른 텍스트 없이 배열만."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"다음 중 적합한 여행지만 선택하세요:\n{json.dumps(enriched, ensure_ascii=False, indent=2)}")
    ])
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        filtered = json.loads(content.strip())
        print(f"필터링 결과: {len(filtered)}개 남음 (원본: {len(enriched)}개)")
        
        return AgentState(
            filtered_options=filtered,
            messages=[f"{len(filtered)}개의 적합한 여행지를 찾았습니다."],
            current_step="filter_options"
        )
    except Exception as e:
        print(f"필터링 오류: {e}")
        return AgentState(
            filtered_options=enriched,
            messages=["필터링을 건너뛰었습니다."],
            current_step="filter_options"
        )

async def rank_destinations_node(state: AgentState) -> AgentState:
    """여행지 순위 매기기 노드"""
    print("\n=== 여행지 순위화 중 ===")
    
    filtered = state["filtered_options"]
    prefs = state["user_preferences"]
    
    system_prompt = f"""사용자에게 가장 적합한 여행지 상위 3개를 선정하세요.
    
    사용자 선호도:
    {json.dumps(prefs, ensure_ascii=False, indent=2)}
    
    각 여행지에 대해:
    - match_score (0-100): 사용자 선호도 일치 점수
    - ranking_reason: 이 순위를 준 구체적 이유
    
    상위 3개를 ranking 순서대로 JSON 배열로 반환하세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"다음 여행지를 순위화하세요:\n{json.dumps(filtered, ensure_ascii=False, indent=2)}")
    ])
    
    try:
        ranked = extract_json(response.content)
        top3 = ranked[:3] if len(ranked) >= 3 else ranked
        print(f"Top {len(top3)} 선정 완료")
        
        return AgentState(
            final_recommendations=top3,
            messages=["최종 추천지를 선정했습니다."],
            current_step="rank_destinations"
        )
    except Exception as e:
        print(f"순위화 오류: {e}")
        return AgentState(
            final_recommendations=filtered[:3],
            messages=["순위화를 건너뛰었습니다."],
            current_step="rank_destinations"
        )


async def final_check_node(state: AgentState) -> AgentState:
    """최종 검증 노드"""
    print("\n=== 최종 검증 중 ===")
    
    recommendations = state["final_recommendations"]
    prefs = state["user_preferences"]
    
    system_prompt = f"""최종 추천이 사용자 요구사항을 충족하는지 확인하세요.
    
    사용자 선호도:
    {json.dumps(prefs, ensure_ascii=False, indent=2)}
    
    추천 결과:
    {json.dumps(recommendations, ensure_ascii=False, indent=2)}
    
    다음을 확인하세요:
    - 추천 이유의 논리성
    - 사용자 니즈 충족도
    - 실현 가능성
    - 정보의 완성도
    
    JSON으로 응답: {{"approved": true/false, "comments": "평가"}}"""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="최종 검증해주세요.")
    ])
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        approved = result.get("approved", True)
        comments = result.get("comments", "")
        
        print(f"최종 검증: {'✓ 통과' if approved else '✗ 실패'}")
        print(f"코멘트: {comments}")
        
        return AgentState(
            messages=[f"최종 검증: {comments}"],
            current_step="final_check"
        )
    except Exception as e:
        print(f"최종 검증 오류: {e}")
        return AgentState(
            messages=["최종 검증을 건너뛰었습니다."],
            current_step="final_check"
        )

async def present_recommendations_node(state: AgentState) -> AgentState:
    """최종 추천 제시 노드"""
    print("\n=== 추천 결과 제시 ===")
    
    recommendations = state["final_recommendations"]
    prefs = state["user_preferences"]
    
    system_prompt = """사용자에게 매력적으로 여행지를 소개하세요.
    
    각 여행지별로:
    1. 🌟 제목과 한 줄 요약
    2. 💡 추천 이유 (사용자 선호도와 연결)
    3. 📅 예상 일정 개요
    4. 💰 예산 가이드
    5. ✨ 핵심 팁 및 주의사항
    
    친근하고 설득력 있게, 이모지를 적절히 사용해서 작성하세요.
    각 여행지를 명확히 구분하고, 읽기 쉽게 포맷팅하세요."""
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
        사용자 선호도: {json.dumps(prefs, ensure_ascii=False)}
        
        추천 여행지: {json.dumps(recommendations, ensure_ascii=False, indent=2)}
        
        매력적인 추천서를 작성해주세요.
        """)
    ])
    
    final_message = f"""
{'='*60}
🌍 당신을 위한 맞춤형 여행지 추천
{'='*60}

{response.content}

{'='*60}
📝 추가 정보가 필요하거나 다른 옵션을 원하시면 말씀해주세요!
{'='*60}
    """
    
    print(final_message)
    
    return AgentState(
        messages=[final_message],
        current_step="present_recommendations"
    )

def increment_retry(state: AgentState) -> AgentState:
    """재시도 카운터 증가"""
    new_count = state.get("retry_count", 0) + 1
    print(f"재시도 카운터 증가: {new_count}")
    return AgentState(
        retry_count=new_count,
        current_step="increment_retry"
    )

def increment_enrich_retry(state: AgentState) -> AgentState:
    """정보 보강 재시도 카운터 증가"""
    new_count = state.get("enrich_retry_count", 0) + 1
    print(f"정보 보강 재시도 카운터 증가: {new_count}")
    return AgentState(
        enrich_retry_count=new_count,
        current_step="increment_enrich_retry"
    )
