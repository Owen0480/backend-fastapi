-- place_photo에 image_url 컬럼 추가 (이미지 URL 저장용). 이미 있으면 에러 나므로 무시.
USE travel;
ALTER TABLE place_photo ADD COLUMN image_url VARCHAR(500) NULL COMMENT '이미지 전체 URL (있으면 API에서 그대로 반환)';
