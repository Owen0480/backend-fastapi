"""
place_photo ↔ place 매칭 검증 및 정리

1) 검증: 같은 photo_file_nm이 서로 다른 visit_area_id(장소)에 붙어 있으면
   이미지-장소 불일치 원인 → 목록 출력
2) 정리: 중복된 이미지당 "한 장소만" 남기고, 장소명이 이미지와 어울리는 쪽 우선
   (해수욕장/해변/바다/펜션 우선, 피자/맛집/식당 제거)

실행: backend-fastapi 폴더에서
  python scripts/verify_and_fix_place_photo.py          # 검증만 (수정 없음)
  python scripts/verify_and_fix_place_photo.py --fix   # 중복 정리 실행 (DELETE)
환경변수: insert_place_data.py와 동일 (.env 사용)

참고: 중복이 없어도 "해변 사진인데 도우개러지로 나온다"면 원본 CSV에서
      해당 이미지가 잘못된 장소에 연결된 경우. 추천 API에서 해변 검색 시
      맛집/피자 계열은 제외하는 로직으로 보완됨.
"""
import os
import sys
from pathlib import Path

# Windows 콘솔 한글 출력
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pymysql
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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


def _place_fit_score(name):
    """장소명이 해변/자연 계열이면 높은 점수, 맛집/피자 계열이면 낮은 점수."""
    if not name:
        return 0
    n = str(name).strip()
    if "해수욕장" in n or "해변" in n or "바다" in n or "해안" in n:
        return 2
    if "펜션" in n or "리조트" in n or "비치" in n:
        return 1
    if any(kw in n for kw in ("피자", "맛집", "식당", "카페", "도우개러지", "음식점")):
        return -1
    return 0


def run_verify(conn, do_fix=False):
    with conn.cursor() as cur:
        # 1) 중복된 photo_file_nm 목록
        cur.execute("""
            SELECT photo_file_nm, COUNT(*) AS cnt
            FROM place_photo
            GROUP BY photo_file_nm
            HAVING COUNT(*) > 1
        """)
        dup_photos = cur.fetchall()
    if not dup_photos:
        print("[검증] 같은 이미지가 여러 장소에 붙은 건 없습니다. (이미지-장소 1:1)")
        return 0

    print(f"[검증] 같은 이미지가 여러 장소에 붙은 건: {len(dup_photos)}개 이미지")
    ids_to_delete = []

    with conn.cursor() as cur:
        for row in dup_photos:
            photo_file_nm = row["photo_file_nm"]
            cur.execute("""
                SELECT pp.id, pp.photo_file_nm, pp.visit_area_id,
                       COALESCE(pp.visit_area_nm, p.visit_area_nm) AS visit_area_nm,
                       p.road_nm_addr
                FROM place_photo pp
                LEFT JOIN place p ON p.visit_area_id = pp.visit_area_id
                WHERE pp.photo_file_nm = %s
            """, (photo_file_nm,))
            rows = cur.fetchall()
            # 점수 높은 순 정렬 → 첫 번째만 남기고 나머지 id 수집
            scored = [(r, _place_fit_score(r.get("visit_area_nm"))) for r in rows]
            scored.sort(key=lambda x: (-x[1], x[0]["id"]))
            keep_id = scored[0][0]["id"]
            for r, score in scored[1:]:
                ids_to_delete.append(r["id"])
            # 로그
            names = [r["visit_area_nm"] or "(없음)" for r in rows]
            print(f"  - {photo_file_nm}: 장소 {names} → 유지: {scored[0][0].get('visit_area_nm')} (id={keep_id})")

    if not ids_to_delete:
        return len(dup_photos)

    if do_fix:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(ids_to_delete))
            cur.execute(f"DELETE FROM place_photo WHERE id IN ({placeholders})", ids_to_delete)
        conn.commit()
        print(f"[정리] 잘못 매칭된 place_photo {len(ids_to_delete)}건 삭제 완료. (같은 이미지당 한 장소만 유지)")
        print("  → FastAPI 서버 재시작 또는 embedding_cache 폴더 삭제 후 재시작하면 추천이 갱신됩니다.")
    else:
        print(f"[정리 대기] --fix 옵션으로 실행하면 위 이미지들에 대해 잘못 붙은 행 {len(ids_to_delete)}건을 삭제합니다.")

    return len(dup_photos)


def run_sample_check(conn, limit=10):
    """place_photo JOIN place 로 이미지-장소-주소 샘플 출력."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pp.photo_file_nm, pp.visit_area_id,
                   COALESCE(pp.visit_area_nm, p.visit_area_nm) AS visit_area_nm,
                   p.road_nm_addr
            FROM place_photo pp
            LEFT JOIN place p ON p.visit_area_id = pp.visit_area_id
            ORDER BY pp.id
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    print("\n[샘플] place_photo ↔ place 매칭 (상위 %d건)" % limit)
    for r in rows:
        print(f"  이미지: {r['photo_file_nm']} | 장소: {r['visit_area_nm']} | 주소: {r['road_nm_addr'] or '-'}")


def run_stats(conn):
    """place_photo / place 건수 및 중복 여부 요약."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM place_photo")
        n_photo = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM place")
        n_place = cur.fetchone()["c"]
        cur.execute("""
            SELECT COUNT(*) AS c FROM (
                SELECT photo_file_nm FROM place_photo GROUP BY photo_file_nm HAVING COUNT(*) > 1
            ) t
        """)
        n_dup = cur.fetchone()["c"]
    print("\n[통계] place_photo %d건, place %d건 | 같은 이미지가 여러 장소에 붙은 이미지 수: %d" % (n_photo, n_place, n_dup))


def main():
    do_fix = "--fix" in sys.argv
    print("MariaDB 연결 중... (이미지-장소 매칭 검증)")
    try:
        conn = _conn()
    except Exception as e:
        print(f"연결 실패: {e}")
        sys.exit(1)
    try:
        run_stats(conn)
        run_verify(conn, do_fix=do_fix)
        run_sample_check(conn, limit=15)
    finally:
        conn.close()
    print("\n완료.")


if __name__ == "__main__":
    main()
