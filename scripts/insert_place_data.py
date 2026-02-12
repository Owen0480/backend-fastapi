"""
국내 여행로그 데이터(수도권/동부권/서부권/제주도 및 도서지역) → place, place_photo 테이블 INSERT

데이터 위치 (프로젝트 루트 = travel_pj 기준):
- 방문지정보: 국내 여행로그 데이터({지역})\Sample\02.라벨링데이터\csv\tn_visit_area_info_방문지정보_{A|B|C|D}.csv
- 관광사진:   국내 여행로그 데이터({지역})\Sample\02.라벨링데이터\csv\tn_tour_photo_관광사진_{A|B|C|D}.csv
- 이미지:     국내 여행로그 데이터({지역})\Sample\01.원천데이터\photo\  (파일명 = PHOTO_FILE_NM)

실행: backend-fastapi 폴더에서
  pip install pymysql pandas   # 한 번만
  python scripts/insert_place_data.py

환경변수(선택):
- DB: MARIADB_HOST, MARIADB_PORT(기본 3306), MARIADB_USER, MARIADB_PASSWORD, MARIADB_DATABASE
- TRAVEL_DATA_ROOT: 국내 여행로그 데이터 상위 폴더 (기본: 프로젝트 루트 = backend-fastapi의 상위)
- STORE_IMAGES=0: image_data에 BLOB 저장 안 함 (image_path만 저장 권장)
- ONLY_EXISTING_IMAGES=1: 실제 이미지 파일이 있는 행만 INSERT
- INSERT_SAMPLE_RATIO=0.1: 지역별 10%만 (수도권/동부권/서부권/제주도 각 10%)
- CLEAR_BEFORE_INSERT=1: 실행 전 place_photo, place 테이블 비우고 넣기
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pymysql
from dotenv import load_dotenv

# .env 로드 (backend-fastapi/.env 에 MARIADB_HOST, MARIADB_PORT 등 있으면 사용)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# backend-fastapi
ROOT = Path(__file__).resolve().parent.parent
# 국내 여행로그 데이터가 있는 프로젝트 루트 (travel_pj)
TRAVEL_DATA_ROOT = Path(os.environ.get("TRAVEL_DATA_ROOT", str(ROOT.parent)))

# 지역명 → CSV/이미지 접미사 (파일명 prefix: a_, b_, c_, d_)
REGIONS = [
    ("수도권", "A"),
    ("동부권", "B"),
    ("서부권", "C"),
    ("제주도 및 도서지역", "D"),
]

STORE_IMAGES = os.environ.get("STORE_IMAGES", "0") == "1"
ONLY_EXISTING_IMAGES = os.environ.get("ONLY_EXISTING_IMAGES", "0") == "1"
# 넣을 데이터 비율 (0.0~1.0). 예: 0.1 이면 지역별 10%씩 INSERT (수도권/동부권/서부권/제주도 균등)
INSERT_SAMPLE_RATIO = float(os.environ.get("INSERT_SAMPLE_RATIO", "1.0"))
if INSERT_SAMPLE_RATIO <= 0 or INSERT_SAMPLE_RATIO > 1:
    INSERT_SAMPLE_RATIO = 1.0
# 1 이면 실행 시 place_photo, place 테이블 비운 뒤 INSERT (DBeaver에서 안 해도 됨)
CLEAR_BEFORE_INSERT = os.environ.get("CLEAR_BEFORE_INSERT", "0") == "1"


def _conn():
    host = os.environ.get("MARIADB_HOST", "localhost")
    port = int(os.environ.get("MARIADB_PORT", "3306"))
    user = os.environ.get("MARIADB_USER", "root")
    password = os.environ.get("MARIADB_PASSWORD", "1234")
    database = os.environ.get("MARIADB_DATABASE", "travel")
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _safe_float(v):
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _safe_str(v, max_len=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    if max_len and len(s) > max_len:
        return s[:max_len]
    return s


def _safe_datetime(v):
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        t = pd.to_datetime(v)
        return t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
    except Exception:
        return None


REGION_TO_SUFFIX = {"수도권": "A", "동부권": "B", "서부권": "C", "제주도 및 도서지역": "D"}


def _get_place_df():
    """4개 지역 방문지정보 CSV 병합, VISIT_AREA_ID 기준 중복 제거"""
    dfs = []
    for region_name, suffix in REGIONS:
        folder = TRAVEL_DATA_ROOT / f"국내 여행로그 데이터({region_name})" / "Sample" / "02.라벨링데이터" / "csv"
        path = folder / f"tn_visit_area_info_방문지정보_{suffix}.csv"
        if not path.exists():
            print(f"  (건너뜀) 없음: {path}")
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            dfs.append(df)
        except Exception as e:
            print(f"  읽기 실패 {path}: {e}")
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["VISIT_AREA_ID"], keep="first")
    return out


def _get_place_df_with_region():
    """지역별 방문지정보 (중복 제거는 지역+ID 기준). 샘플 시 접두어(A,B,C,D)로 지역 구분용."""
    dfs = []
    for region_name, suffix in REGIONS:
        folder = TRAVEL_DATA_ROOT / f"국내 여행로그 데이터({region_name})" / "Sample" / "02.라벨링데이터" / "csv"
        path = folder / f"tn_visit_area_info_방문지정보_{suffix}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df["_region"] = region_name
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["_region", "VISIT_AREA_ID"], keep="first")
    return out


def _get_photo_df_and_place_lookup():
    """4개 지역 관광사진 CSV 병합 + 지역별 이미지 경로 매핑. (photo_df, visit_area_nm_map, region_photo_dirs)"""
    photo_rows = []
    visit_area_nm_map = {}  # (region_name, visit_area_id) -> VISIT_AREA_NM from 방문지정보
    region_photo_dirs = {}  # region_name -> Path(photo dir)

    for region_name, suffix in REGIONS:
        base = TRAVEL_DATA_ROOT / f"국내 여행로그 데이터({region_name})" / "Sample"
        label_csv = base / "02.라벨링데이터" / "csv" / f"tn_tour_photo_관광사진_{suffix}.csv"
        visit_csv = base / "02.라벨링데이터" / "csv" / f"tn_visit_area_info_방문지정보_{suffix}.csv"
        photo_dir = base / "01.원천데이터" / "photo"

        if not label_csv.exists():
            print(f"  (건너뜀) 없음: {label_csv}")
            continue
        region_photo_dirs[region_name] = photo_dir

        # 방문지정보에서 VISIT_AREA_ID -> VISIT_AREA_NM
        if visit_csv.exists():
            try:
                vdf = pd.read_csv(visit_csv, encoding="utf-8-sig")
                for _, r in vdf.iterrows():
                    vid = _safe_str(r.get("VISIT_AREA_ID"), 20)
                    vnm = _safe_str(r.get("VISIT_AREA_NM"), 200)
                    if vid:
                        visit_area_nm_map[(region_name, vid)] = vnm
            except Exception as e:
                print(f"  방문지정보 읽기 실패 {visit_csv}: {e}")

        try:
            df = pd.read_csv(label_csv, encoding="utf-8-sig")
            df["_region"] = region_name
            photo_rows.append(df)
        except Exception as e:
            print(f"  읽기 실패 {label_csv}: {e}")

    if not photo_rows:
        return pd.DataFrame(), visit_area_nm_map, region_photo_dirs
    photo_df = pd.concat(photo_rows, ignore_index=True)
    return photo_df, visit_area_nm_map, region_photo_dirs


def insert_place(conn, place_df=None, id_column="VISIT_AREA_ID"):
    """국내 여행로그 데이터 방문지정보 → place 테이블. id_column 있으면 그 컬럼을 visit_area_id로 사용 (지역 접두어용)."""
    df = place_df if place_df is not None else _get_place_df()
    if df.empty:
        print("place: 읽을 CSV 없음 (국내 여행로그 데이터 방문지정보)")
        return 0
    inserted = 0
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            visit_area_id = _safe_str(row.get(id_column) or row.get("VISIT_AREA_ID"), 30)
            if not visit_area_id:
                continue
            try:
                cur.execute(
                    """
                    INSERT INTO place
                    (visit_area_id, visit_area_nm, road_nm_addr, lotno_addr, x_coord, y_coord, sgg_cd)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        visit_area_id,
                        _safe_str(row.get("VISIT_AREA_NM"), 200),
                        _safe_str(row.get("ROAD_NM_ADDR"), 500),
                        _safe_str(row.get("LOTNO_ADDR"), 500),
                        _safe_float(row.get("X_COORD")),
                        _safe_float(row.get("Y_COORD")),
                        _safe_str(row.get("SGG_CD"), 20),
                    ),
                )
                inserted += 1
            except pymysql.IntegrityError as e:
                if e.args[0] == 1062:
                    pass
                else:
                    raise
        conn.commit()
    print(f"place: {inserted}건 INSERT")
    return inserted


def insert_place_photo(conn, photo_df=None, visit_area_nm_map=None, region_photo_dirs=None, id_column=None):
    """국내 여행로그 데이터 관광사진 + 이미지 경로 → place_photo 테이블. id_column 있으면 그 값을 visit_area_id로 사용 (지역 접두어용)."""
    if photo_df is None or visit_area_nm_map is None or region_photo_dirs is None:
        photo_df, visit_area_nm_map, region_photo_dirs = _get_photo_df_and_place_lookup()
    if photo_df.empty:
        print("place_photo: 읽을 CSV 없음 (국내 여행로그 데이터 관광사진)")
        return 0

    # image_path 컬럼 존재 여부 확인 (있으면 상대 경로 저장)
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM place_photo LIKE 'image_path'")
        has_image_path = cur.fetchone() is not None

    inserted = 0
    with conn.cursor() as cur:
        for _, row in photo_df.iterrows():
            photo_file_nm = _safe_str(row.get("PHOTO_FILE_NM"), 255)
            if not photo_file_nm:
                continue
            region_name = row.get("_region", "")
            photo_dir = region_photo_dirs.get(region_name)
            img_path = (photo_dir / photo_file_nm) if photo_dir else None
            if img_path is None:
                img_path = Path()
            else:
                img_path = img_path.resolve()
            has_image_file = img_path.exists() if img_path else False

            if ONLY_EXISTING_IMAGES and not has_image_file:
                continue

            visit_area_id = _safe_str(row.get(id_column) if id_column else row.get("VISIT_AREA_ID"), 30) or ""
            visit_area_nm = visit_area_nm_map.get((region_name, visit_area_id)) or _safe_str(row.get("VISIT_AREA_NM"), 200)

            image_data = None
            if STORE_IMAGES and has_image_file:
                try:
                    with open(img_path, "rb") as f:
                        image_data = f.read()
                except Exception as e:
                    print(f"  이미지 읽기 실패 {img_path}: {e}")

            # 상대 경로: 프로젝트 루트 기준 (예: 국내 여행로그 데이터(수도권)/Sample/01.원천데이터/photo/xxx.jpg)
            image_path_rel = None
            if has_image_path and region_name and photo_dir and has_image_file:
                try:
                    image_path_rel = f"국내 여행로그 데이터({region_name})/Sample/01.원천데이터/photo/{photo_file_nm}"
                except Exception:
                    pass

            if has_image_path:
                try:
                    cur.execute(
                        """
                        INSERT INTO place_photo
                        (photo_file_nm, visit_area_id, visit_area_nm, image_data, image_path, photo_file_frmat, photo_file_dt, x_coord, y_coord)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            photo_file_nm,
                            visit_area_id,
                            visit_area_nm,
                            image_data,
                            image_path_rel,
                            _safe_str(row.get("PHOTO_FILE_FRMAT"), 10),
                            _safe_datetime(row.get("PHOTO_FILE_DT")),
                            _safe_float(row.get("PHOTO_FILE_X_COORD")),
                            _safe_float(row.get("PHOTO_FILE_Y_COORD")),
                        ),
                    )
                    inserted += 1
                except pymysql.ProgrammingError as e:
                    if 1054 in e.args or "Unknown column" in str(e):
                        # image_path 컬럼 없음 → 기존 스키마로 INSERT
                        cur.execute(
                            """
                            INSERT INTO place_photo
                            (photo_file_nm, visit_area_id, visit_area_nm, image_data, photo_file_frmat, photo_file_dt, x_coord, y_coord)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                photo_file_nm,
                                visit_area_id,
                                visit_area_nm,
                                image_data,
                                _safe_str(row.get("PHOTO_FILE_FRMAT"), 10),
                                _safe_datetime(row.get("PHOTO_FILE_DT")),
                                _safe_float(row.get("PHOTO_FILE_X_COORD")),
                                _safe_float(row.get("PHOTO_FILE_Y_COORD")),
                            ),
                        )
                        inserted += 1
                    else:
                        raise
                except pymysql.IntegrityError:
                    pass
            else:
                try:
                    cur.execute(
                        """
                        INSERT INTO place_photo
                        (photo_file_nm, visit_area_id, visit_area_nm, image_data, photo_file_frmat, photo_file_dt, x_coord, y_coord)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            photo_file_nm,
                            visit_area_id,
                            visit_area_nm,
                            image_data,
                            _safe_str(row.get("PHOTO_FILE_FRMAT"), 10),
                            _safe_datetime(row.get("PHOTO_FILE_DT")),
                            _safe_float(row.get("PHOTO_FILE_X_COORD")),
                            _safe_float(row.get("PHOTO_FILE_Y_COORD")),
                        ),
                    )
                    inserted += 1
                except pymysql.IntegrityError:
                    pass
        conn.commit()
    print(f"place_photo: {inserted}건 INSERT")
    return inserted


def _clear_tables(conn):
    """place_photo, place 테이블 비우기 (외래키 순서대로)"""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM place_photo")
        cur.execute("DELETE FROM place")
        conn.commit()
    print("place_photo, place 테이블 비움.")


def main():
    print(f"데이터 루트: {TRAVEL_DATA_ROOT}")
    if INSERT_SAMPLE_RATIO < 1.0:
        print(f"샘플 비율: 지역별 {INSERT_SAMPLE_RATIO*100:.0f}% (수도권/동부권/서부권/제주도)")
    if CLEAR_BEFORE_INSERT:
        print("CLEAR_BEFORE_INSERT=1 → 실행 전 DB 비움.")
    print("MariaDB 연결 중...")
    try:
        conn = _conn()
    except Exception as e:
        print(f"연결 실패: {e}")
        print("  환경변수: MARIADB_HOST, MARIADB_PORT, MARIADB_USER, MARIADB_PASSWORD, MARIADB_DATABASE")
        sys.exit(1)
    try:
        if CLEAR_BEFORE_INSERT:
            _clear_tables(conn)
        if INSERT_SAMPLE_RATIO < 1.0:
            photo_df, visit_area_nm_map, region_photo_dirs = _get_photo_df_and_place_lookup()
            if photo_df.empty:
                print("place_photo CSV 없음. 샘플링 건너뜀.")
                insert_place(conn)
                insert_place_photo(conn)
            else:
                # 지역별로 동일 비율 샘플 (수도권/동부권/서부권/제주도 균등)
                photo_sampled = photo_df.groupby("_region", group_keys=True).apply(
                    lambda g: g.sample(frac=INSERT_SAMPLE_RATIO, random_state=42)
                ).reset_index(level=0)  # _region 컬럼 유지
                # 지역별 ID 접두어(A,B,C,D)로 동일 ID 충돌 방지 (월정리해수욕장 vs 수원 등 주소 섞임 방지)
                photo_sampled = photo_sampled.copy()
                photo_sampled["visit_area_id_prefixed"] = photo_sampled["_region"].map(REGION_TO_SUFFIX) + "_" + photo_sampled["VISIT_AREA_ID"].astype(str)
                needed_visit_ids = set(photo_sampled["visit_area_id_prefixed"].unique())
                place_df = _get_place_df_with_region()
                place_df = place_df.copy()
                place_df["visit_area_id_prefixed"] = place_df["_region"].map(REGION_TO_SUFFIX) + "_" + place_df["VISIT_AREA_ID"].astype(str)
                place_filtered = place_df[place_df["visit_area_id_prefixed"].isin(needed_visit_ids)] if not place_df.empty else place_df
                region_counts = photo_sampled.groupby("_region").size()
                print(f"place: {len(needed_visit_ids)}개 장소, place_photo: {len(photo_sampled)}건 (지역별 {INSERT_SAMPLE_RATIO*100:.0f}%, ID 접두어 A/B/C/D 적용)")
                for r, c in region_counts.items():
                    print(f"  - {r}: {c}건")
                insert_place(conn, place_filtered, id_column="visit_area_id_prefixed")
                insert_place_photo(conn, photo_sampled, visit_area_nm_map, region_photo_dirs, id_column="visit_area_id_prefixed")
        else:
            insert_place(conn)
            insert_place_photo(conn)
    finally:
        conn.close()
    print("완료.")


if __name__ == "__main__":
    main()
