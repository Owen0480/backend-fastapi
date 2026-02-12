from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
from app.core.config import settings
from app.api.v1.router import api_router
from app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """브라우저 favicon 요청 시 404 방지 (빈 응답)."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/")
def root():
    """루트 접속 시 데모 페이지 링크를 보여줍니다."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8"><title>Backend FastAPI</title></head>
    <body style="font-family:Malgun Gothic;margin:24px;">
    <h1>Welcome to FastAPI Project</h1>
    <p><a href="/demo" style="font-size:1.2em;">▶ 여행 데모 페이지 열기</a></p>
    <p>또는 <a href="/api/v1/demo/travel-demo">/api/v1/demo/travel-demo</a></p>
    <p><a href="/docs">API 문서 (Swagger)</a></p>
    </body></html>
    """)


@app.get("/demo")
def demo_redirect():
    """/demo 접속 시 데모 페이지로 리다이렉트 (URL 단순화)."""
    return RedirectResponse(url=f"{settings.API_V1_STR}/demo/travel-demo", status_code=302)


# =============================================================================
# 정적 파일 / 이미지 (로컬 없으면 국내 여행로그 데이터 폴더에서 서빙)

base_path = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(base_path, "images")
travel_data_root = os.path.abspath(os.environ.get("TRAVEL_DATA_ROOT") or os.path.join(base_path, ".."))
_travel_regions = ["수도권", "동부권", "서부권", "제주도 및 도서지역"]


def _resolve_image_path(filename: str):
    """backend-fastapi/images 또는 국내 여행로그 데이터(지역)/Sample/01.원천데이터/photo 에서 파일 경로 반환."""
    if ".." in filename or "/" in filename.replace("\\", "/"):
        return None
    local = os.path.join(images_path, filename)
    if os.path.isfile(local):
        return local
    for region in _travel_regions:
        path = os.path.join(travel_data_root, f"국내 여행로그 데이터({region})", "Sample", "01.원천데이터", "photo", filename)
        if os.path.isfile(path):
            return path
    return None


@app.get("/images/{file_path:path}")
def serve_image(file_path: str):
    """이미지 파일 서빙 (로컬 images/ 또는 국내 여행로그 데이터 photo 폴더)."""
    resolved = _resolve_image_path(file_path.strip())
    if resolved:
        mt = "image/png" if file_path.lower().endswith(".png") else "image/jpeg"
        return FileResponse(resolved, media_type=mt)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.get("/home")
def get_home():
    """프론트 이미지 검색 페이지 (정적 HTML)."""
    html_path = os.path.join(base_path, "..", "travel-frontend", "public", "imageSeaching.html")
    if not os.path.exists(html_path):
        return {"error": f"HTML 파일을 찾을 수 없습니다: {html_path}"}
    return FileResponse(html_path)


@app.get("/image-search")
def get_image_search():
    """이미지 검색 페이지 (정적 HTML). /home 과 동일."""
    return get_home()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
