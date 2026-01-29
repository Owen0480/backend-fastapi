from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend FastAPI"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    OPENAI_API_KEY: str = "AIzaSyAjkr1eefz0UhXpy7ZkPxw8RXh8-a6DqHo"
    GOOGLE_API_KEY: Optional[str] = None  # Gemini. 없으면 여행 분류는 키워드/기본값으로 동작
    
    # 타임아웃 (초). LLM·그래프 호출이 이 시간을 넘기면 중단하고 에러 표출
    LLM_TIMEOUT_SEC: int = 25
    GRAPH_TIMEOUT_SEC: int = 60
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
