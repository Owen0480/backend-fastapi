from fastapi import APIRouter, UploadFile, File, HTTPException
# 우리가 만든 서비스를 불러옵니다.
from app.services.recommend_service import recommend_service 

router = APIRouter()

@router.post("/analyze")
async def analyze_travel_image(file: UploadFile = File(...)):
    """
    사용자가 올린 사진을 받아 추천 서비스를 실행하는 통로입니다.
    """
    try:
        # 1. 사용자가 보낸 파일을 바이트 데이터로 읽기
        contents = await file.read()
        
        # 2. recommend_service의 분석 함수 호출
        result = recommend_service.analyze_image_bytes(contents)
        
        return result
        
    except Exception as e:
        # 에러 발생 시 500 에러 반환
        raise HTTPException(status_code=500, detail=str(e))