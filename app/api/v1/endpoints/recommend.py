from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.recommend import RecommendAnalyzeResponse
from app.services.recommend_service import recommend_service

router = APIRouter()


@router.post("/analyze", response_model=RecommendAnalyzeResponse)
async def analyze_travel_image(file: UploadFile = File(...)):
    """
    사용자가 올린 사진을 받아 추천 서비스를 실행합니다.
    성공 시: success, count, results[] (place_name, address, score, image_file, guide)
    실패 시: success, ai_analysis
    """
    try:
        contents = await file.read()
        result = recommend_service.analyze_image(contents)
        return RecommendAnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))