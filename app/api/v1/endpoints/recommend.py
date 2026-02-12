from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Request

from app.schemas.recommend import RecommendAnalyzeResponse
from app.services.recommend_service import recommend_service

router = APIRouter()


def _ensure_image_url(result: dict, base_url: str) -> dict:
    """result에 image_url 없으면 base_url + /images/ + image_file 로 채움."""
    if result.get("image_url"):
        return result
    base = base_url.rstrip("/")
    result["image_url"] = f"{base}/images/{result.get('image_file', '')}"
    return result


@router.post("/analyze", response_model=RecommendAnalyzeResponse)
async def analyze_travel_image(
    request: Request,
    file: UploadFile = File(...),
    preference: str = Form(default=""),
):
    """
    사용자가 올린 사진을 받아 추천 서비스를 실행합니다.
    성공 시: success, count, results[] (place_name, address, score, image_file, image_url, guide)
    이미지는 image_url(전체 URL)로 사용.
    """
    try:
        contents = await file.read()
        result = recommend_service.analyze_image(contents, preference=preference or "")
        if result.get("results"):
            base_url = str(request.base_url)
            result["results"] = [_ensure_image_url(r, base_url) for r in result["results"]]
        return RecommendAnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))