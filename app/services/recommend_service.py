import os
import io
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class RecommendService:
    def __init__(self):
        # 1. 모델 로드
        self.model = SentenceTransformer('clip-ViT-B-32')
        
        # 2. Gemini 설정
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.gemini_model = None

        # 3. 데이터 및 DB 이미지 로드
        self.merged_df = self._load_csv_data()
        self.db_images_folder = 'images'
        
        # 4. [최적화] DB 이미지 임베딩 미리 계산 (Caching)
        self.db_features = []
        self.db_filenames = []
        self._precompute_db_embeddings()

    def _load_csv_data(self):
        """CSV 데이터 로드 및 병합 (backend-fastapi/data/ 기준)"""
        # backend-fastapi 폴더 기준 경로 (app/services/ 에서 두 단계 상위)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        tour_path = os.path.join(base_dir, 'data', 'tour.csv')   # PHOTO_FILE_NM 있음
        place_path = os.path.join(base_dir, 'data', 'place.csv')  # VISIT_AREA_NM, ROAD_NM_ADDR 있음
        try:
            if not os.path.exists(tour_path) or not os.path.exists(place_path):
                print(f"데이터 파일 없음: tour={tour_path}, place={place_path}")
                return pd.DataFrame()
            photo_df = pd.read_csv(tour_path, encoding='utf-8-sig')
            place_df = pd.read_csv(place_path, encoding='utf-8-sig')
            merged = pd.merge(photo_df, place_df, on='VISIT_AREA_ID', how='inner')
            if 'PHOTO_FILE_NM' in merged.columns:
                merged['PHOTO_FILE_NM'] = merged['PHOTO_FILE_NM'].astype(str).str.strip()
            return merged
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return pd.DataFrame()

    def _precompute_db_embeddings(self):
        """서버 시작 시 images 폴더 내 모든 사진의 특징값을 미리 추출"""
        if not os.path.exists(self.db_images_folder):
            print(f"경고: {self.db_images_folder} 폴더가 없습니다.")
            return

        print("DB 이미지 분석 중... (이 작업은 처음에 한 번만 실행됩니다)")
        files = [f for f in os.listdir(self.db_images_folder) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for f in files:
            try:
                img_path = os.path.join(self.db_images_folder, f)
                emb = self.model.encode(Image.open(img_path))
                self.db_features.append(emb)
                self.db_filenames.append(f)
            except Exception as e:
                print(f"파일 처리 실패 ({f}): {e}")
        
        if self.db_features:
            self.db_features = np.array(self.db_features)
            print(f"총 {len(self.db_filenames)}개의 이미지 분석 완료.")

    def analyze_image(self, image_input):
        """
        사용자 이미지를 받아 유사 장소를 추천합니다.
        image_input: 이미지 바이트 스트림 또는 PIL Image 객체
        """
        # 1. 사용자 이미지 임베딩
        if isinstance(image_input, bytes):
            user_img = Image.open(io.BytesIO(image_input))
        else:
            user_img = image_input
            
        user_img_emb = self.model.encode(user_img)

        # 2. 유사도 계산 (벡터 연산으로 고속 처리)
        if len(self.db_features) == 0:
            return {"success": False, "message": "비교할 DB 이미지가 없습니다."}

        cos_scores = util.cos_sim(user_img_emb, self.db_features)[0]
        k = min(3, len(self.db_features))
        top_results = np.argpartition(-cos_scores, range(k))[:k]  # 상위 k개

        results = []
        for idx in top_results:
            score = cos_scores[idx].item()
            file_name = self.db_filenames[idx]

            if score > 0.6:  # 임계값
                place_info = self._get_place_info(file_name)
                if place_info['success']:
                    results.append({
                        "place_name": place_info['place_name'],
                        "address": place_info['address'],
                        "score": score,
                        "image_file": file_name,
                        # guide는 Top-1만 Gemini로 생성하고, 나머지는 CSV 기반 짧은 문구로 채움
                        "guide": "",
                        # CSV 기반 짧은 설명을 위한 부가 정보
                        "poi_name": place_info.get("poi_name", ""),
                        "visit_area_type_cd": place_info.get("visit_area_type_cd", ""),
                        "residence_time_min": place_info.get("residence_time_min", ""),
                        "dgstfn": place_info.get("dgstfn", ""),
                    })

        if results:
            # 점수 높은 순으로 정렬
            results.sort(key=lambda x: x['score'], reverse=True)
            # Top-1만 Gemini 호출 (429/한도 문제 최소화)
            top1 = results[0]
            top1_guide = self._generate_travel_guide(top1.get("place_name", ""), top1.get("address", ""))
            if not top1_guide:
                top1_guide = self._generate_short_guide(
                    place_name=top1.get("place_name", ""),
                    address=top1.get("address", ""),
                    score=top1.get("score", 0),
                    poi_name=top1.get("poi_name", ""),
                    visit_area_type_cd=top1.get("visit_area_type_cd", ""),
                    residence_time_min=top1.get("residence_time_min", ""),
                    dgstfn=top1.get("dgstfn", ""),
                )
            results[0]["guide"] = top1_guide

            # Top-2/3는 CSV 기반 짧은 설명으로 채움
            for i in range(1, len(results)):
                r = results[i]
                r["guide"] = self._generate_short_guide(
                    place_name=r.get("place_name", ""),
                    address=r.get("address", ""),
                    score=r.get("score", 0),
                    poi_name=r.get("poi_name", ""),
                    visit_area_type_cd=r.get("visit_area_type_cd", ""),
                    residence_time_min=r.get("residence_time_min", ""),
                    dgstfn=r.get("dgstfn", ""),
                )

            # 응답에는 프론트가 쓰는 필드만 남김 (부가 필드는 제거)
            for r in results:
                r.pop("poi_name", None)
                r.pop("visit_area_type_cd", None)
                r.pop("residence_time_min", None)
                r.pop("dgstfn", None)
            return {"success": True, "count": len(results), "results": results}
        else:
            # 유사 장소 없을 시 Gemini로 설명 시도 (429/한도 초과 시 500 방지)
            if self.gemini_model:
                try:
                    analysis = self.gemini_model.generate_content(["이 사진이 어떤 사진인지 한국어로 한 문장 설명해줘.", user_img])
                    ai_text = (analysis.text if analysis and hasattr(analysis, "text") else "") or ""
                    return {"success": False, "ai_analysis": ai_text or "유사한 여행지를 찾지 못했습니다."}
                except Exception as e:
                    err_msg = str(e).lower()
                    if "429" in err_msg or "quota" in err_msg or "rate" in err_msg:
                        return {"success": False, "ai_analysis": "유사한 여행지를 찾지 못했습니다. (AI 설명 한도 초과 — 잠시 후 다시 시도해 주세요.)"}
                    return {"success": False, "ai_analysis": "유사한 여행지를 찾지 못했습니다. 다른 사진을 올려 보세요."}
            return {"success": False, "ai_analysis": "유사한 여행지를 찾지 못했습니다. 다른 사진을 올려 보세요."}

    def _safe_str(self, val, default=""):
        """NaN/None을 제거하고 항상 str로 반환 (Pydantic 검증 통과용)"""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        s = str(val).strip()
        return s if s else default

    def _get_place_info(self, file_name):
        """파일명으로 CSV에서 장소 정보 검색"""
        if self.merged_df.empty or 'PHOTO_FILE_NM' not in self.merged_df.columns:
            return {"success": False}
        match = self.merged_df[self.merged_df['PHOTO_FILE_NM'] == file_name]
        if not match.empty:
            row = match.iloc[0]
            # merge 시 place 쪽이 _y, tour 쪽이 _x → 장소명은 place(VISIT_AREA_NM_y) 우선
            p_name = self._safe_str(
                row.get('VISIT_AREA_NM_y') or row.get('VISIT_AREA_NM_x') or row.get('VISIT_AREA_NM'),
                '장소명 없음'
            )
            # 도로명 우선, 없으면 지번 주소
            addr = self._safe_str(row.get('ROAD_NM_ADDR')) or self._safe_str(row.get('LOTNO_ADDR'), '주소 정보 없음')
            poi_name = self._safe_str(row.get("POI_NM")) or self._safe_str(row.get("POI_NM_y")) or self._safe_str(row.get("POI_NM_x"))
            visit_area_type_cd = self._safe_str(row.get("VISIT_AREA_TYPE_CD"))
            residence_time_min = self._safe_str(row.get("RESIDENCE_TIME_MIN"))
            dgstfn = self._safe_str(row.get("DGSTFN"))
            return {
                "place_name": p_name,
                "address": addr,
                "poi_name": poi_name,
                "visit_area_type_cd": visit_area_type_cd,
                "residence_time_min": residence_time_min,
                "dgstfn": dgstfn,
                "success": True,
            }
        return {"success": False}

    def _generate_short_guide(
        self,
        place_name: str,
        address: str,
        score: float,
        poi_name: str = "",
        visit_area_type_cd: str = "",
        residence_time_min: str = "",
        dgstfn: str = "",
    ) -> str:
        """LLM 없이 CSV 기반으로 짧은 안내 문구 생성"""
        parts = []
        if poi_name:
            parts.append(f"POI: {poi_name}")
        if visit_area_type_cd:
            parts.append(f"유형코드: {visit_area_type_cd}")
        if residence_time_min:
            parts.append(f"권장 체류: {residence_time_min}분")
        if dgstfn:
            parts.append(f"만족도: {dgstfn}")
        meta = " · ".join(parts)
        pct = int(round((score or 0) * 100))
        base = f"{pct}% 유사한 분위기의 후보 여행지입니다."
        if meta:
            base += f" ({meta})"
        if address:
            base += f" 주소: {address}"
        if place_name:
            return f"{place_name} — {base}"
        return base

    def _generate_travel_guide(self, place_name, address):
        """Gemini 여행 가이드 생성"""
        if not self.gemini_model:
            return "가이드를 생성할 수 없습니다. (.env에 GEMINI_API_KEY 설정 후 서버 재시작)"
        prompt = f"당신은 전문 여행 가이드입니다. {place_name}(주소: {address})에 대해 한 줄 요약, 핵심 특징, 여행 팁/맛집을 5문장 이내로 아주 간결하게 한국어로 알려주세요."
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else f"{place_name} 방문을 추천합니다. 주소: {address}"
        except Exception as e:
            # API 한도/키 오류 시에도 화면에는 안내 문구 표시
            return f"{place_name}({address}) 방문을 추천합니다. 상세 가이드는 Gemini API 설정을 확인해 주세요."

# 싱글톤 인스턴스 생성
recommend_service = RecommendService()