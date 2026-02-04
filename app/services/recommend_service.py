import os
import io
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb

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

        # 5. (정석 RAG용) 텍스트 임베딩 + Chroma 인덱스 준비
        # - 장소 문서(장소명/주소/POI/만족도)를 청크로 만들어 벡터DB에 저장
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.chroma_dir = os.path.join(base_dir, "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name="place_docs")
        self._ensure_place_docs_index()

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

    def _ensure_place_docs_index(self):
        """Chroma에 장소 문서가 없으면 CSV 기반으로 생성/저장."""
        try:
            existing = self.chroma_collection.count()
        except Exception:
            existing = 0
        if existing and existing > 0:
            return

        if self.merged_df.empty:
            print("경고: merged_df가 비어 있어 Chroma 인덱스를 만들 수 없습니다.")
            return

        print("Chroma 장소 문서 인덱싱 중... (처음 1회)")

        def chunk_text(text: str, size: int = 450) -> list[str]:
            text = (text or "").strip()
            if not text:
                return []
            return [text[i:i + size] for i in range(0, len(text), size)]

        # VISIT_AREA_ID 기준으로 대표 행(첫 행) 추출
        grouped = self.merged_df.groupby("VISIT_AREA_ID", sort=False)
        ids = []
        docs = []
        metas = []
        for visit_area_id, g in grouped:
            row = g.iloc[0]
            place_name = str(row.get("VISIT_AREA_NM_y") or row.get("VISIT_AREA_NM_x") or row.get("VISIT_AREA_NM") or "").strip()
            address = str(row.get("ROAD_NM_ADDR") or row.get("LOTNO_ADDR") or "").strip()
            poi = str(row.get("POI_NM") or row.get("POI_NM_y") or row.get("POI_NM_x") or "").strip()
            dgstfn = str(row.get("DGSTFN") or "").strip()

            doc = "\n".join([
                f"장소명: {place_name or '정보 없음'}",
                f"주소: {address or '정보 없음'}",
                f"POI: {poi or '정보 없음'}",
                f"만족도: {dgstfn or '정보 없음'}",
            ])

            for ci, chunk in enumerate(chunk_text(doc, 450)):
                ids.append(f"{visit_area_id}_{ci}")
                docs.append(chunk)
                metas.append({
                    "visit_area_id": str(visit_area_id),
                    "place_name": place_name or "",
                    "source": "csv",
                    "chunk_index": ci,
                })

        if not ids:
            print("경고: Chroma에 넣을 장소 문서가 없습니다.")
            return

        # 임베딩 생성 후 upsert
        embeddings = self.text_model.encode(docs).tolist()
        self.chroma_collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        print(f"Chroma 인덱싱 완료: {len(ids)} chunks")

    def _retrieve_place_chunks(self, visit_area_id: str, query: str, top_k: int = 3) -> list[str]:
        """특정 장소(visit_area_id) 범위에서 관련 청크 Top-k 검색."""
        if not query:
            query = ""
        try:
            q_emb = self.text_model.encode([query]).tolist()[0]
            res = self.chroma_collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where={"visit_area_id": str(visit_area_id)},
                include=["documents"],
            )
            docs = (res.get("documents") or [[]])[0]
            return [d for d in docs if d]
        except Exception:
            return []

    def analyze_image(self, image_input, preference: str = ""):
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
            # Chroma에서 (사진요약 대신) preference + 장소명 기반으로 관련 청크를 retrieval
            visit_area_id = top1.get("visit_area_id", "")
            retrieval_query = " ".join([preference or "", top1.get("place_name", ""), top1.get("poi_name", "")]).strip()
            retrieved_chunks = self._retrieve_place_chunks(visit_area_id, retrieval_query, top_k=3) if visit_area_id else []
            top1_guide = self._generate_travel_guide(
                place_name=top1.get("place_name", ""),
                address=top1.get("address", ""),
                poi_name=top1.get("poi_name", ""),
                dgstfn=top1.get("dgstfn", ""),
                preference=preference or "",
                retrieved_chunks=retrieved_chunks,
            )
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
                r.pop("visit_area_id", None)
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
                    reason = self._gemini_error_reason(e)
                    return {"success": False, "ai_analysis": f"유사한 여행지를 찾지 못했습니다. [실패 사유] {reason}"}
            return {"success": False, "ai_analysis": "유사한 여행지를 찾지 못했습니다. 다른 사진을 올려 보세요."}

    def _gemini_error_reason(self, e: Exception) -> str:
        """Gemini API 예외를 사용자용 한글 사유로 변환"""
        err_msg = (str(e).strip() or "알 수 없는 오류").lower()
        if "429" in err_msg or "quota" in err_msg or "exceeded" in err_msg:
            return "Gemini API 할당량 초과 — 요금제/결제 확인 후 잠시 후 재시도해 주세요."
        if "billing" in err_msg or "billable" in err_msg:
            return "결제 정보가 설정되지 않았습니다. Google AI Studio에서 결제를 활성화해 주세요."
        if "api key" in err_msg or "invalid" in err_msg or "401" in err_msg:
            return "API 키가 잘못되었거나 만료되었습니다. .env의 GEMINI_API_KEY를 확인해 주세요."
        return str(e).strip()[:120] or "알 수 없는 오류"

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
                "visit_area_id": self._safe_str(row.get("VISIT_AREA_ID")),
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

    def _generate_travel_guide(self, place_name, address, poi_name="", dgstfn="", preference: str = "", retrieved_chunks: list[str] | None = None):
        """Gemini 여행 가이드 생성 (근거 + Chroma retrieval 기반)"""
        if not self.gemini_model:
            return "가이드를 생성할 수 없습니다. (.env에 GEMINI_API_KEY 설정 후 서버 재시작)"
        retrieved_chunks = retrieved_chunks or []
        evidence_lines = [
            f"장소명: {place_name or '정보 없음'}",
            f"주소: {address or '정보 없음'}",
            f"POI: {poi_name or '정보 없음'}",
            f"만족도: {dgstfn or '정보 없음'}",
        ]
        evidence = "\n".join(evidence_lines)
        retrieved_text = "\n\n".join([f"- {c}" for c in retrieved_chunks]) if retrieved_chunks else "- (검색 근거 없음)"
        pref = (preference or "").strip()
        prompt = f"""너는 여행 가이드야. 아래 '근거'에 있는 정보만 사용해서 답해.
근거에 없는 내용은 절대 지어내지 말고, 해당 항목은 쓰지 말고 건너뛰어.

[사용자 취향]
{pref or '없음'}

[근거]
{evidence}

[검색된 문서 청크]
{retrieved_text}

[요청]
취향이 있으면 반드시 그 관점에서 써줘. (예: 가족 → 가족 여행에 어울리는 점, 맛집 → 먹거리·맛집 탐방 관점, 힐링 → 휴식·힐링 관점)
1) 한 줄 요약 (20자 내외) — 취향에 맞춰 관점을 잡고, 장소명·주소·POI·만족도 중 있는 정보로만 작성. 취향이 "없음"이면 일반 요약.
2) 핵심 특징 1개 — 취향 관점에서 이 장소의 강점을 한 줄로. (같은 근거라도 취향이 바뀌면 다른 문장으로 써줘.)
3) 여행 팁·맛집 — 근거에 해당 정보가 있을 때만 1개씩 적고, 없으면 "3)" 항목 전체를 생략.

전체는 5문장 이내, 한국어로 아주 간결하게.
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else f"{place_name} 방문을 추천합니다. 주소: {address}"
        except Exception as e:
            reason = self._gemini_error_reason(e)
            return f"{place_name}({address}) 방문을 추천합니다. [가이드 생성 실패] {reason}"

# 싱글톤 인스턴스 생성
recommend_service = RecommendService()