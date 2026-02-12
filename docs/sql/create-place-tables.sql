-- place, place_photo 테이블 생성 (MariaDB travel DB)
-- 실행: HeidiSQL/DBeaver/MySQL Workbench 등에서 travel DB 선택 후 실행

USE travel;

-- 장소 테이블
CREATE TABLE IF NOT EXISTS place (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    visit_area_id VARCHAR(20) NOT NULL UNIQUE,
    visit_area_nm VARCHAR(200),
    road_nm_addr VARCHAR(500),
    lotno_addr VARCHAR(500),
    x_coord DOUBLE,
    y_coord DOUBLE,
    sgg_cd VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 장소-사진 테이블 (이미지 BLOB 또는 image_path로 경로만 저장)
CREATE TABLE IF NOT EXISTS place_photo (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    photo_file_nm VARCHAR(255) NOT NULL,
    visit_area_id VARCHAR(20) NOT NULL,
    visit_area_nm VARCHAR(200),
    image_data LONGBLOB,
    image_path VARCHAR(500) COMMENT '국내 여행로그 데이터 기준 상대 경로. image_data 없을 때 이 경로에서 로드',
    photo_file_frmat VARCHAR(10),
    photo_file_dt DATETIME,
    x_coord DOUBLE,
    y_coord DOUBLE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 조회 속도용 인덱스 (선택)
CREATE INDEX idx_place_photo_visit_area_id ON place_photo(visit_area_id);
CREATE INDEX idx_place_photo_photo_file_nm ON place_photo(photo_file_nm);
