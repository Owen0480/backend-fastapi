from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse
from app.core.config import settings
from app.api.v1.router import api_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(api_router, prefix=settings.API_V1_STR)


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

#============================================================================

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_path = os.path.join(base_path, "images")

from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory=images_path), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/home")
def get_home():
    """프론트 이미지 검색 페이지 (정적 HTML)."""
    html_path = os.path.join(base_path, "..", "travel-frontend", "public", "imageSeaching.html")
    if not os.path.exists(html_path):
        return {"error": f"HTML 파일을 찾을 수 없습니다: {html_path}"}
    return FileResponse(html_path)

#============================================================================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)  # reload=True : 코드 변경 시 서버 자동 재시작