from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI LLM Project"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    OPENAI_API_KEY: str = "your-api-key-here"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
