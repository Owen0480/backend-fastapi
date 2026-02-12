import os
import io
import json
import hashlib
from collections import OrderedDict
import pandas as pd
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb

load_dotenv()

try:
    from app.core.config import settings
except Exception:
    settings = None


def _get_mariadb_conn():
    """이미지 추천용 MariaDB(3306) 연결. 설정 없으면 None."""
    host = getattr(settings, "MARIADB_HOST", None) or os.getenv("MARIADB_HOST")
    if not host:
        return None
    try:
        import pymysql
        port = getattr(settings, "MARIADB_PORT", 3306) if settings else int(os.getenv("MARIADB_PORT", "3306"))
        user = getattr(settings, "MARIADB_USER", "root") if settings else os.getenv("MARIADB_USER", "root")
        password = getattr(settings, "MARIADB_PASSWORD", "1234") if settings else os.getenv("MARIADB_PASSWORD", "1234")
        database = getattr(settings, "MARIADB_DATABASE", "travel") if settings else os.getenv("MARIADB_DATABASE", "travel")
        return pymysql.connect(
            host=host, port=port, user=user, password=password, database=database,
            charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as e:
        print(f"MariaDB(이미지 추천) 연결 실패: {e}")
        return None


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

        # 3. 데이터 로드: 3306 도커 DB 우선, 없으면 CSV
        self._db_conn = _get_mariadb_conn()
        if self._db_conn:
            self.merged_df = self._load_from_db()
            self.use_recommend_db = not self.merged_df.empty
        else:
            self.merged_df = pd.DataFrame()
            self.use_recommend_db = False
        if self.merged_df.empty:
            self.merged_df = self._load_csv_data()
        self.db_images_folder = 'images'
        # 국내 여행로그 데이터 루트 (image_path 상대 경로 해석용)
        self._base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.travel_data_root = os.path.abspath(os.environ.get("TRAVEL_DATA_ROOT") or os.path.join(self._base_dir, ".."))
        # image_path 없을 때 국내 여행로그 데이터 폴더에서 찾기 위한 지역 목록
        self._travel_data_regions = ["수도권", "동부권", "서부권", "제주도 및 도서지역"]
        # 업로드 이미지 임베딩 캐시 (같은 사진 재업로드 시 재사용, 최대 100개)
        self._upload_embedding_cache = OrderedDict()
        self._upload_embedding_cache_max = 100
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

    def _load_from_db(self):
        """이미지 추천용 MariaDB(3306)에서 place + place_photo 로드 → merged_df 형태"""
        if not self._db_conn:
            return pd.DataFrame()
        try:
            with self._db_conn.cursor() as cur:
                try:
                    cur.execute("""
                        SELECT pp.photo_file_nm AS PHOTO_FILE_NM, pp.visit_area_id AS VISIT_AREA_ID,
                               COALESCE(pp.visit_area_nm, p.visit_area_nm) AS VISIT_AREA_NM,
                               p.road_nm_addr AS ROAD_NM_ADDR, p.lotno_addr AS LOTNO_ADDR,
                               pp.image_url AS image_url
                        FROM place_photo pp
                        LEFT JOIN place p ON p.visit_area_id = pp.visit_area_id
                    """)
                except Exception:
                    cur.execute("""
                        SELECT pp.photo_file_nm AS PHOTO_FILE_NM, pp.visit_area_id AS VISIT_AREA_ID,
                               COALESCE(pp.visit_area_nm, p.visit_area_nm) AS VISIT_AREA_NM,
                               p.road_nm_addr AS ROAD_NM_ADDR, p.lotno_addr AS LOTNO_ADDR
                        FROM place_photo pp
                        LEFT JOIN place p ON p.visit_area_id = pp.visit_area_id
                    """)
                rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["PHOTO_FILE_NM"] = df["PHOTO_FILE_NM"].astype(str).str.strip()
            print(f"이미지 추천 DB(3306) 로드: place_photo {len(df)}건")
            return df
        except Exception as e:
            print(f"DB 로드 실패: {e}")
            return pd.DataFrame()

    def _embedding_cache_dir(self):
        """임베딩 캐시 저장 경로"""
        return os.path.join(self._base_dir, "embedding_cache")

    def _load_embedding_cache(self, expected_count, source="db"):
        """캐시가 있고 건수가 맞으면 (features, filenames) 반환. source: 'db' | 'images_folder'"""
        cache_dir = self._embedding_cache_dir()
        meta_path = os.path.join(cache_dir, "meta.json")
        features_path = os.path.join(cache_dir, "features.npy")
        filenames_path = os.path.join(cache_dir, "filenames.json")
        if not os.path.isfile(meta_path) or not os.path.isfile(features_path) or not os.path.isfile(filenames_path):
            return None, None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if source == "db":
                if meta.get("place_photo_count") != expected_count:
                    return None, None
            else:
                if meta.get("source") != "images_folder" or meta.get("file_count") != expected_count:
                    return None, None
            features = np.load(features_path)
            with open(filenames_path, "r", encoding="utf-8") as f:
                filenames = json.load(f)
            if len(filenames) != features.shape[0]:
                return None, None
            # DB 모드에서만: 구버전 캐시(파일명만)면 이미지-장소 매칭 오류 가능 → 재계산 유도
            if source == "db" and filenames and not any(isinstance(x, str) and "|" in x for x in filenames[: min(10, len(filenames))]):
                print("임베딩 캐시가 구버전(파일명만)이라 이미지-장소 매칭을 위해 재계산합니다.")
                return None, None
            return features, filenames
        except Exception as e:
            print(f"임베딩 캐시 로드 실패: {e}")
            return None, None

    def _save_embedding_cache(self, count, source="db"):
        """현재 db_features, db_filenames를 캐시에 저장. source: 'db' | 'images_folder'"""
        if len(self.db_filenames) == 0:
            return
        cache_dir = self._embedding_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        try:
            np.save(os.path.join(cache_dir, "features.npy"), self.db_features)
            with open(os.path.join(cache_dir, "filenames.json"), "w", encoding="utf-8") as f:
                json.dump(self.db_filenames, f, ensure_ascii=False)
            meta = {"place_photo_count": count} if source == "db" else {"source": "images_folder", "file_count": count}
            with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            print(f"임베딩 캐시 저장 완료: {cache_dir} ({len(self.db_filenames)}개) — 다음부터 캐시 로드로 빠르게 시작")
        except Exception as e:
            print(f"임베딩 캐시 저장 실패: {e}")

    def _precompute_db_embeddings(self):
        """서버 시작 시 이미지 특징값 미리 추출. 캐시 있으면 로드, 없으면 계산 후 저장."""
        if self.use_recommend_db and self._db_conn and not self.merged_df.empty:
            try:
                with self._db_conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) AS cnt FROM place_photo")
                    row = cur.fetchone()
                    place_photo_count = (row or {}).get("cnt") or 0
                features, filenames = self._load_embedding_cache(place_photo_count)
                if features is not None and filenames is not None:
                    self.db_features = features
                    self.db_filenames = filenames
                    print(f"임베딩 캐시 로드: {len(self.db_filenames)}개 (DB {place_photo_count}건 기준) — 분석 생략")
                    return
            except Exception as e:
                print(f"캐시 확인 실패: {e}")
            # 캐시 없음 또는 건수 불일치 → DB에서 이미지 읽어서 계산
            print(f"임베딩 캐시 없음 또는 건수 불일치 → DB 이미지 분석 시작 (한 번만 실행, 끝나면 캐시 저장)")
            print(f"DB 이미지 분석 중... (3306 place_photo 기준, 이미지 루트: {self.travel_data_root})")
            try:
                with self._db_conn.cursor() as cur:
                    try:
                        cur.execute("SELECT visit_area_id, photo_file_nm, image_data, image_path FROM place_photo")
                    except Exception:
                        try:
                            cur.execute("SELECT visit_area_id, photo_file_nm, image_data FROM place_photo")
                        except Exception:
                            cur.execute("SELECT photo_file_nm, image_data, image_path FROM place_photo")
                    rows = cur.fetchall()
                total = len(rows)
                report_every = max(100, total // 20)  # 최대 20번 진행률 출력
                for idx, row in enumerate(rows):
                    if (idx + 1) % report_every == 0 or (idx + 1) == total:
                        done = len(self.db_filenames)
                        pct = (idx + 1) * 100 // total if total else 0
                        print(f"이미지 분석 중... {idx + 1}/{total} ({pct}%) → 성공 {done}건")
                    f = (row.get("photo_file_nm") or "").strip()
                    if not f or not (f.endswith((".jpg", ".png", ".jpeg"))):
                        continue
                    vid = (row.get("visit_area_id") or "").strip() if row.get("visit_area_id") is not None else ""
                    # 이미지-장소 정확 매칭: 같은 파일명이 다른 장소에 있으면 구분하기 위해 key 사용
                    store_key = f"{vid}|{f}" if vid else f
                    try:
                        img = None
                        blob = row.get("image_data")
                        if blob:
                            img = Image.open(io.BytesIO(blob))
                        else:
                            # image_path(국내 여행로그 데이터 상대경로) 우선, 없으면 국내 여행로그 데이터 폴더에서 찾기, 마지막으로 images/
                            path = None
                            rel_path = (row.get("image_path") or "").strip()
                            if rel_path and os.path.isabs(rel_path):
                                path = rel_path
                            elif rel_path:
                                path = os.path.normpath(os.path.join(self.travel_data_root, rel_path))
                            if (not path or not os.path.exists(path)) and getattr(self, "_travel_data_regions", None):
                                # image_path 없거나 파일 없으면 국내 여행로그 데이터(수도권) 등 4개 폴더에서 파일명으로 검색
                                for region in self._travel_data_regions:
                                    candidate = os.path.join(
                                        self.travel_data_root,
                                        f"국내 여행로그 데이터({region})",
                                        "Sample", "01.원천데이터", "photo", f
                                    )
                                    if os.path.exists(candidate):
                                        path = candidate
                                        break
                            if (not path or not os.path.exists(path)):
                                path = os.path.join(self._base_dir, self.db_images_folder, f)
                            if path and os.path.exists(path):
                                img = Image.open(path)
                        if img is not None:
                            emb = self.model.encode(img)
                            self.db_features.append(emb)
                            self.db_filenames.append(store_key)
                    except Exception as e:
                        print(f"파일 처리 실패 ({f}): {e}")
            except Exception as e:
                print(f"DB 이미지 조회 실패: {e}")
            if self.db_features:
                self.db_features = np.array(self.db_features)
                print(f"총 {len(self.db_filenames)}개의 이미지 분석 완료 (DB 기준).")
                print("임베딩 캐시 저장 중... (완료될 때까지 서버 중단/파일 저장하지 마세요)")
                try:
                    with self._db_conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) AS cnt FROM place_photo")
                        n = (cur.fetchone() or {}).get("cnt") or 0
                    self._save_embedding_cache(n)
                except Exception as e:
                    print(f"임베딩 캐시 저장 단계 오류: {e}")
            return

        if not os.path.exists(self.db_images_folder):
            print(f"경고: {self.db_images_folder} 폴더가 없습니다.")
            return

        files = [f for f in os.listdir(self.db_images_folder)
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
        total_files = len(files)
        if total_files == 0:
            return
        # images 폴더 모드에서도 캐시 사용: 있으면 로드 후 생략
        features, filenames = self._load_embedding_cache(total_files, source="images_folder")
        if features is not None and filenames is not None:
            self.db_features = features
            self.db_filenames = filenames
            print(f"임베딩 캐시 로드: {len(self.db_filenames)}개 (images 폴더 기준) — 분석 생략")
            return
        print(f"DB 이미지 분석 중... (총 {total_files}개, 이 작업은 처음에 한 번만 실행됩니다)")
        report_every = max(50, total_files // 15)
        for i, f in enumerate(files):
            if (i + 1) % report_every == 0 or (i + 1) == total_files:
                pct = (i + 1) * 100 // total_files if total_files else 0
                print(f"이미지 분석 중... {i + 1}/{total_files} ({pct}%)")
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
            self._save_embedding_cache(total_files, source="images_folder")

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
        # 1. 사용자 이미지 임베딩 (같은 이미지면 캐시 재사용)
        user_img_emb = None
        if isinstance(image_input, bytes):
            img_hash = hashlib.sha256(image_input).hexdigest()
            if img_hash in self._upload_embedding_cache:
                user_img_emb = self._upload_embedding_cache[img_hash].copy()
                self._upload_embedding_cache.move_to_end(img_hash)  # 최근 사용으로
            if user_img_emb is None:
                user_img = Image.open(io.BytesIO(image_input))
                user_img_emb = self.model.encode(user_img)
                self._upload_embedding_cache[img_hash] = user_img_emb.copy()
                self._upload_embedding_cache.move_to_end(img_hash)
                while len(self._upload_embedding_cache) > self._upload_embedding_cache_max:
                    self._upload_embedding_cache.popitem(last=False)
        else:
            user_img = image_input
            user_img_emb = self.model.encode(user_img)

        # 2. 유사도 계산 (벡터 연산으로 고속 처리)
        if len(self.db_features) == 0:
            return {"success": False, "message": "비교할 DB 이미지가 없습니다."}

        cos_scores = util.cos_sim(user_img_emb, self.db_features)[0]
        # 같은 이미지가 여러 장소에 붙어 있을 수 있으므로 후보를 넉넉히 뽑은 뒤, 이미지별로 한 장소만 남김
        k_candidates = min(20, len(self.db_features))
        top_candidates = np.argpartition(-cos_scores, range(k_candidates))[:k_candidates]

        raw_results = []
        for idx in top_candidates:
            score = cos_scores[idx].item()
            key = self.db_filenames[idx]
            if "|" in key:
                _, file_name = key.split("|", 1)
            else:
                file_name = key

            if score <= 0.6:
                continue
            place_info = self._get_place_info(key)
            if not place_info.get('success'):
                continue
            place_name = (place_info.get('place_name') or "").strip()
            r = {
                "place_name": place_name,
                "address": place_info.get('address', ''),
                "score": score,
                "image_file": file_name,
                "guide": "",
                "poi_name": place_info.get("poi_name", ""),
                "visit_area_type_cd": place_info.get("visit_area_type_cd", ""),
                "residence_time_min": place_info.get("residence_time_min", ""),
                "dgstfn": place_info.get("dgstfn", ""),
                "visit_area_id": place_info.get("visit_area_id", ""),
            }
            if place_info.get("image_url"):
                r["image_url"] = place_info.get("image_url")
            raw_results.append(r)

        # 같은 이미지 파일에 여러 장소가 붙은 경우: 이미지와 어울리는 장소명(해수욕장·해변 등) 우선, 하나만 노출
        def _image_place_fit_score(name):
            if not name:
                return 0
            n = name
            if "해수욕장" in n or "해변" in n or "바다" in n or "해안" in n:
                return 2
            if "펜션" in n or "리조트" in n or "비치" in n:
                return 1
            return 0

        def _is_restaurant_place(name):
            if not name:
                return False
            n = name
            return any(kw in n for kw in ("피자", "맛집", "식당", "카페", "음식점", "도우개러지", "빵", "커피"))

        # 상위 후보가 대부분 해변/해수욕장 계열이면 → 맛집/피자 계열은 잘못 매칭된 데이터로 보고 제외
        beach_like_count = sum(1 for r in raw_results if _image_place_fit_score(r["place_name"]) >= 1)
        search_looks_beach = beach_like_count >= 2
        if search_looks_beach:
            raw_results = [r for r in raw_results if not _is_restaurant_place(r["place_name"])]

        seen_image = {}
        for r in raw_results:
            f = r["image_file"]
            fit = _image_place_fit_score(r["place_name"])
            if f not in seen_image or (fit > seen_image[f][1]) or (fit == seen_image[f][1] and r["score"] > seen_image[f][0]["score"]):
                seen_image[f] = (r, fit)
        results = [v[0] for v in seen_image.values()]
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:3]

        if results:
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

    def _get_place_info(self, file_name_or_key):
        """파일명 또는 key(visit_area_id|photo_file_nm)로 장소 정보 검색. key 사용 시 이미지-장소 1:1 매칭."""
        if self.merged_df.empty or 'PHOTO_FILE_NM' not in self.merged_df.columns:
            return {"success": False}
        if "|" in file_name_or_key:
            visit_area_id, photo_file_nm = file_name_or_key.split("|", 1)
            match = self.merged_df[
                (self.merged_df['VISIT_AREA_ID'].astype(str) == str(visit_area_id)) &
                (self.merged_df['PHOTO_FILE_NM'] == photo_file_nm)
            ]
        else:
            match = self.merged_df[self.merged_df['PHOTO_FILE_NM'] == file_name_or_key]
        if not match.empty:
            row = match.iloc[0]
            # merge 시 place 쪽이 _y, tour 쪽이 _x → 장소명은 place(VISIT_AREA_NM_y) 우선
            p_name = self._safe_str(
                row.get('VISIT_AREA_NM_y') or row.get('VISIT_AREA_NM_x') or row.get('VISIT_AREA_NM'),
                ''
            )
            # 도로명 우선, 없으면 지번 주소 (없으면 빈 문자열로 노출 안 함)
            addr = self._safe_str(row.get('ROAD_NM_ADDR')) or self._safe_str(row.get('LOTNO_ADDR'), '')
            poi_name = self._safe_str(row.get("POI_NM")) or self._safe_str(row.get("POI_NM_y")) or self._safe_str(row.get("POI_NM_x"))
            visit_area_type_cd = self._safe_str(row.get("VISIT_AREA_TYPE_CD"))
            residence_time_min = self._safe_str(row.get("RESIDENCE_TIME_MIN"))
            dgstfn = self._safe_str(row.get("DGSTFN"))
            out = {
                "visit_area_id": self._safe_str(row.get("VISIT_AREA_ID")),
                "place_name": p_name,
                "address": addr,
                "poi_name": poi_name,
                "visit_area_type_cd": visit_area_type_cd,
                "residence_time_min": residence_time_min,
                "dgstfn": dgstfn,
                "success": True,
            }
            if "image_url" in row and row.get("image_url"):
                out["image_url"] = self._safe_str(row.get("image_url"), "")
            return out
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
        prompt = f"""너는 친근한 여행 가이드야. 아래 [근거]와 [검색된 문서]만 참고해서, 지어내지 말고 있는 정보만 써줘.

[사용자 취향] {pref or '없음'}

[근거]
{evidence}

[검색된 문서]
{retrieved_text}

[작성 요청]
- 취향이 있으면 그 관점(가족/맛집/힐링 등)에 맞춰 자연스럽게 써줘.
- 형식: 1) 한 줄 요약 2) 이 장소만의 특징 한 줄 3) 여행 팁(근거에 있을 때만). 없으면 3) 생략.
- 2) 특징에는 주소·위치·도로명을 쓰지 말고, 이 장소의 분위기·매력·특색만 간략히 써줘. (주소는 [근거]에 이미 있음)
- 말투: 친구에게 추천하듯 부드럽고 자연스럽게. 딱딱한 설명체 말고 구어체에 가깝게.
- 5문장 이내, 한국어.
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else f"{place_name} 방문을 추천합니다. 주소: {address}"
        except Exception as e:
            reason = self._gemini_error_reason(e)
            return f"{place_name}({address}) 방문을 추천합니다. [가이드 생성 실패] {reason}"

# 싱글톤 인스턴스 생성
recommend_service = RecommendService() 