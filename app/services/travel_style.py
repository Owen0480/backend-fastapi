"""
여행 스타일 서비스
선택한 관심사에 따라 타입별 퍼센트(복합) 분석, Gemini API로 여행지 추천
"""
import json
import logging
import requests

from app.core.config import settings

try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False
from app.domain.travel_style import TRAVEL_TYPES, KEYWORD_TO_TYPE

logger = logging.getLogger(__name__)

# === Gemini API (config에서 키 사용) ===
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS_ORDER = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.0-flash",
    "gemini-pro",
]

def _compute_type_breakdown(interests: list[str]) -> list[dict]:
    """관심사 3개 → 각 타입별 퍼센트 (선택한 것에 따라 복합적으로)"""
    type_counts: dict[str, int] = {}
    for kw in interests:
        t = KEYWORD_TO_TYPE.get(kw) or "복합형"
        type_counts[t] = type_counts.get(t, 0) + 1
    n = len(interests)
    breakdown = [
        {"type": t, "percent": round(count / n * 100), "count": count}
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1])
    ]
    return breakdown


def _composite_label(breakdown: list[dict]) -> str:
    """퍼센트 높은 순으로 타입명만 이어붙이기 (숫자 없음)"""
    if not breakdown:
        return "복합형"
    if len(breakdown) == 1 and breakdown[0]["percent"] == 100:
        return breakdown[0]["type"]
    return "·".join(b["type"] for b in breakdown)


_TYPE_SHORT_DESC = {
    "액티브형": "활동적인 여행",
    "힐링형": "휴식과 힐링",
    "문화형": "문화·예술",
    "자연형": "자연 감상",
    "미식형": "맛집·음식",
    "모험형": "도전과 모험",
    "감성형": "분위기와 감성",
    "복합형": "다양한 경험",
}


def _composite_description(breakdown: list[dict]) -> str:
    """복합 타입에 대한 설명 (퍼센트 높은 순 타입들 조합)"""
    if not breakdown:
        return TRAVEL_TYPES["복합형"]["description"]
    if len(breakdown) == 1 and breakdown[0]["percent"] == 100:
        return TRAVEL_TYPES.get(breakdown[0]["type"], TRAVEL_TYPES["복합형"])["description"]
    parts = [_TYPE_SHORT_DESC.get(b["type"], b["type"]) for b in breakdown]
    return ", ".join(parts) + "를 골고루 즐기는 타입입니다."


def _fallback_analyze(interests: list[str]) -> dict:
    """Ollama 없을 때 domain KEYWORD_TO_TYPE 기반 퍼센트 분석 (여행지 추천은 AI 필수 → 빈 배열)"""
    breakdown = _compute_type_breakdown(interests)
    primary_type = breakdown[0]["type"] if breakdown else "복합형"
    type_info = dict(TRAVEL_TYPES.get(primary_type, TRAVEL_TYPES["복합형"]))
    type_info["destinations"] = []
    type_info["description"] = _composite_description(breakdown)
    return {
        "matched_type": _composite_label(breakdown),
        "confidence": 85,
        "reason": f"선택한 관심사({', '.join(interests)})를 기반으로 타입별 비율을 분석했습니다. "
        "추천 여행지는 AI로 생성됩니다. (Gemini API 연결 후 다시 시도해 주세요)",
        "secondary_type": None,
        "type_info": type_info,
        "user_interests": interests,
    }


def analyze_interests(interests: list[str]) -> dict:
    """
    관심사 3개를 받아 Gemini API로 여행 타입을 매칭합니다.
    """
    types_description = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TRAVEL_TYPES.items()
    ])

    breakdown = _compute_type_breakdown(interests)
    type_order = "·".join(b["type"] for b in breakdown) if breakdown else "복합형"

    prompt = f"""
다음은 미리 정의된 여행 타입들입니다:

{types_description}

사용자가 선택한 관심사: {', '.join(interests)}
(분석 결과: {type_order} - 퍼센트 높은 순)

**필수**: 선택한 관심사에 맞는 **한국 내 여행지 2~3곳**을 AI가 직접 추천해주세요.
여러 타입이 섞인 경우, 그 조합에 맞는 여행지를 추천하세요. (예: 액티브+미식+힐링 → 활동·맛집·휴식이 모두 있는 곳)

반드시 다음 JSON 형식으로만 응답해주세요:
{{
    "matched_type": "복합형",
    "confidence": 95,
    "reason": "선택한 관심사를 기반으로 한 분석 이유 (2-3문장)",
    "secondary_type": null,
    "destinations": [
        {{"name": "여행지1", "desc": "한 줄 설명 (선택 관심사와 연결)"}},
        {{"name": "여행지2", "desc": "한 줄 설명"}},
        {{"name": "여행지3", "desc": "한 줄 설명"}}
    ]
}}

규칙: destinations는 반드시 2~3개, 한국 내 여행지만, 선택한 관심사 조합에 맞게 추천.
"""
    key = (settings.GOOGLE_API_KEY or "").strip()  # config/.env 에서 Gemini 키 불러옴
    if not key:
        logger.warning("GOOGLE_API_KEY 없음. config 또는 .env 에 Gemini API 키를 넣으면 AI 추천이 구동됩니다.")
        return _fallback_analyze(interests)

    # 1) google-generativeai 라이브러리로 시도
    if _HAS_GENAI:
        for model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.3,
                            response_mime_type="application/json",
                        ),
                    )
                except Exception:
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(temperature=0.3),
                    )
                llm_response = (response.text or "").strip()
                if not llm_response:
                    continue
                try:
                    result = json.loads(llm_response)
                except json.JSONDecodeError:
                    if "```json" in llm_response:
                        llm_response = llm_response.split("```json")[1].split("```")[0]
                    result = json.loads(llm_response.strip())
                breakdown = _compute_type_breakdown(interests)
                result["matched_type"] = _composite_label(breakdown)
                primary_type = breakdown[0]["type"] if breakdown else "복합형"
                type_info = dict(TRAVEL_TYPES.get(primary_type, TRAVEL_TYPES["복합형"]))
                type_info["description"] = _composite_description(breakdown)
                type_info["destinations"] = result.get("destinations") or []
                result["type_info"] = type_info
                result["user_interests"] = interests
                return result
            except Exception as e:
                logger.warning("Gemini SDK [%s] 실패: %s", model_name, e)
                continue

    # 2) REST API 직접 호출
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
    }
    payloads = [
        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}},
        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.3}},
    ]

    for model in GEMINI_MODELS_ORDER:
        url = f"{GEMINI_BASE}/{model}:generateContent?key={key}"
        for payload in payloads:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if not response.ok:
                    err_body = (response.text or "")[:400]
                    logger.warning("Gemini [%s] HTTP %s: %s", model, response.status_code, err_body)
                    continue
                data = response.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    logger.warning("Gemini [%s] 응답에 candidates 없음", model)
                    continue
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if not parts or "text" not in parts[0]:
                    continue
                llm_response = parts[0]["text"]
                try:
                    result = json.loads(llm_response)
                except json.JSONDecodeError:
                    if "```json" in llm_response:
                        llm_response = llm_response.split("```json")[1].split("```")[0]
                    result = json.loads(llm_response.strip())

                breakdown = _compute_type_breakdown(interests)
                result["matched_type"] = _composite_label(breakdown)
                primary_type = breakdown[0]["type"] if breakdown else "복합형"
                type_info = dict(TRAVEL_TYPES.get(primary_type, TRAVEL_TYPES["복합형"]))
                type_info["description"] = _composite_description(breakdown)
                type_info["destinations"] = result.get("destinations") or []
                result["type_info"] = type_info
                result["user_interests"] = interests
                return result
            except Exception as e:
                logger.warning("Gemini [%s] 예외: %s", model, e)
                continue

    return _fallback_analyze(interests)
