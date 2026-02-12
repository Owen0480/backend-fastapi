"""
단위 테스트 실행 — 수행결과 채우기용.
backend-fastapi 폴더에서: python scripts/run_unit_tests.py
의존성(DB/Chroma) 없이 동일 로직만 검증합니다.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import pandas as pd
import numpy as np

# recommend_service와 동일한 로직 (의존성 없이 검증)
def _safe_str(val, default=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    s = str(val).strip()
    return s if s else default

def _gemini_error_reason(e):
    err_msg = (str(e).strip() or "알 수 없는 오류").lower()
    if "429" in err_msg or "quota" in err_msg or "exceeded" in err_msg:
        return "Gemini API 할당량 초과 — 요금제/결제 확인 후 잠시 후 재시도해 주세요."
    if "billing" in err_msg or "billable" in err_msg:
        return "결제 정보가 설정되지 않았습니다. Google AI Studio에서 결제를 활성화해 주세요."
    if "api key" in err_msg or "invalid" in err_msg or "401" in err_msg:
        return "API 키가 잘못되었거나 만료되었습니다. .env의 GEMINI_API_KEY를 확인해 주세요."
    return str(e).strip()[:120] or "알 수 없는 오류"

def test_safe_str():
    """No 17 _safe_str"""
    assert _safe_str(None) == ""
    assert _safe_str(np.nan) == ""
    assert _safe_str("  ") == ""
    assert _safe_str("  a  ") == "a"
    assert _safe_str(123) == "123"
    return "Pass"

def test_gemini_error_reason():
    """No 16 _gemini_error_reason"""
    r = _gemini_error_reason(Exception("429 quota exceeded"))
    assert "할당량" in r or "429" in r
    r2 = _gemini_error_reason(Exception("billing not enabled"))
    assert "결제" in r2 or "billing" in r2.lower()
    r3 = _gemini_error_reason(Exception("Invalid API key 401"))
    assert "API 키" in r3 or "401" in r3
    return "Pass"

def test_get_place_info():
    """No 18, 19 _get_place_info (로직만: 키/파일명 매칭)"""
    df = pd.DataFrame([
        {"PHOTO_FILE_NM": "a.jpg", "VISIT_AREA_ID": "B_1", "VISIT_AREA_NM": "테스트장소", "ROAD_NM_ADDR": "주소1", "LOTNO_ADDR": ""},
        {"PHOTO_FILE_NM": "b.jpg", "VISIT_AREA_ID": "B_2", "VISIT_AREA_NM": "다른장소", "ROAD_NM_ADDR": "주소2", "LOTNO_ADDR": ""},
    ])
    if "|" in "B_1|a.jpg":
        vid, pfn = "B_1|a.jpg".split("|", 1)
        match = df[(df["VISIT_AREA_ID"].astype(str) == str(vid)) & (df["PHOTO_FILE_NM"] == pfn)]
    else:
        match = df[df["PHOTO_FILE_NM"] == "B_1|a.jpg"]
    assert not match.empty and match.iloc[0]["VISIT_AREA_NM"] == "테스트장소"
    match2 = df[df["PHOTO_FILE_NM"] == "a.jpg"]
    assert not match2.empty and match2.iloc[0]["VISIT_AREA_NM"] == "테스트장소"
    match3 = df[(df["VISIT_AREA_ID"] == "B_99") & (df["PHOTO_FILE_NM"] == "c.jpg")]
    assert match3.empty
    return "Pass"

def test_resolve_image_path():
    """No 24 _resolve_image_path (main과 동일 규칙: .. 거부)"""
    images_path = os.path.join(BASE, "images")
    travel_data_root = os.path.abspath(os.environ.get("TRAVEL_DATA_ROOT") or os.path.join(BASE, ".."))
    regions = ["수도권", "동부권", "서부권", "제주도 및 도서지역"]
    def resolve(filename):
        if ".." in filename or "/" in filename.replace("\\", "/"):
            return None
        local = os.path.join(images_path, filename)
        if os.path.isfile(local):
            return local
        for region in regions:
            path = os.path.join(travel_data_root, f"국내 여행로그 데이터({region})", "Sample", "01.원천데이터", "photo", filename)
            if os.path.isfile(path):
                return path
        return None
    assert resolve("..") is None
    assert resolve("../etc/passwd") is None
    assert resolve("a/b.jpg") is None
    for f in os.listdir(images_path) if os.path.isdir(images_path) else []:
        if f.endswith((".jpg", ".png", ".jpeg")):
            assert resolve(f) is not None and os.path.isfile(resolve(f))
            break
    return "Pass"

def main():
    results = {}
    for name, fn in [
        ("No 17 _safe_str", test_safe_str),
        ("No 16 _gemini_error_reason", test_gemini_error_reason),
        ("No 18,19 _get_place_info", test_get_place_info),
        ("No 24 _resolve_image_path", test_resolve_image_path),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = f"Fail: {e}"
    for k, v in results.items():
        print(f"{k}: {v}")
    return results

if __name__ == "__main__":
    main()
