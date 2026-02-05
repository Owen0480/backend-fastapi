from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend FastAPI"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    OPENAI_API_KEY: str = "AIzaSyAjkr1eefz0UhXpy7ZkPxw8RXh8-a6DqHo"
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # 타임아웃 (초). LLM·그래프 호출이 이 시간을 넘기면 중단하고 에러 표출
    LLM_TIMEOUT_SEC: int = 25
    GRAPH_TIMEOUT_SEC: int = 60
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore" # .env에 정의된 다른 변수들이 있어도 에러 방지


settings = Settings()
