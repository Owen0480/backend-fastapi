import os
import pandas as pd
from PIL import Image
import io
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class RecommendService:
    def __init__(self):
        # 1. 모델 로드 (서버 시작 시 한 번만 실행)
        self.model = SentenceTransformer('clip-ViT-B-32')
        
        # 2. Gemini 설정
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 3. CSV 데이터 로드 및 병합
        self.merged_df = self._load_csv_data()
        
        # 4. DB 이미지 경로 설정
        self.db_images_folder = 'images' 

    def _load_csv_data(self):
        """내부용: CSV 데이터 로드 및 전처리"""
        photo_path = 'data/place.csv'
        place_path = 'data/tour.csv'
        
        photo_df = pd.read_csv(photo_path, encoding='utf-8-sig')
        place_df = pd.read_csv(place_path, encoding='utf-8-sig')
        
        merged = pd.merge(photo_df, place_df, on='VISIT_AREA_ID', how='inner')
        if 'PHOTO_FILE_NM' in merged.columns:
            merged['PHOTO_FILE_NM'] = merged['PHOTO_FILE_NM'].astype(str).str.strip()
        return merged

    def analyze_image_bytes(self, image_bytes):
        """FastAPI 라우터에서 바이트 데이터를 받아 분석 수행"""
        user_img = Image.open(io.BytesIO(image_bytes))
        user_img_emb = self.model.encode(user_img)
        
        db_image_files = [f for f in os.listdir(self.db_images_folder) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if not db_image_files:
            return {"success": False, "message": "DB 이미지가 없습니다."}

        best_score = -1
        best_match_file = ""

        # 이미지 비교 루프
        for file_name in db_image_files:
            db_img_path = os.path.join(self.db_images_folder, file_name)
            try:
                db_img_emb = self.model.encode(Image.open(db_img_path))
                score = util.cos_sim(user_img_emb, db_img_emb).item()
                if score > best_score:
                    best_score = score
                    best_match_file = file_name
            except:
                continue

        # 결과 판단 (유사도 60% 기준)
        if best_score > 0.6:
            place_info = self._get_place_info(best_match_file)
            if place_info['success']:
                guide = self._generate_travel_guide(place_info['place_name'], place_info['address'])
                return {
                    "success": True,
                    "best_score": best_score,
                    "place_name": place_info['place_name'],
                    "address": place_info['address'],
                    "best_match_file": best_match_file,
                    "guide": guide
                }
        
        # 유사도가 낮을 경우 Gemini의 직접 분석 결과 반환
        analysis_result = self.gemini_model.generate_content(["이 사진이 어떤 사진인지 한 문장으로 설명해줘.", user_img])
        return {"success": False, "best_score": best_score, "ai_analysis": analysis_result.text}

    def _get_place_info(self, best_match_file):
        """내부용: 파일명으로 CSV에서 정보 추출"""
        normalized_name = str(best_match_file).strip()
        match_info = self.merged_df[self.merged_df['PHOTO_FILE_NM'] == normalized_name].copy()
        
        if not match_info.empty:
            row = match_info.iloc[0]
            place_name = row.get('VISIT_AREA_NM_x') or row.get('VISIT_AREA_NM')
            address = row.get('ROAD_NM_ADDR', '주소 정보 없음')
            return {"place_name": place_name, "address": address, "success": True}
        return {"success": False}

    def _generate_travel_guide(self, place_name, address):
        """내부용: Gemini 가이드 생성"""
        prompt = f"당신은 전문 가이드입니다. {place_name}(주소: {address})에 대해 한 줄 요약, 특징, 맛집 정보를 5문장 이내로 아주 간결하게 한국어로 알려주세요."
        response = self.gemini_model.generate_content(prompt)
        return response.text

# 싱글톤 객체 생성
recommend_service = RecommendService()