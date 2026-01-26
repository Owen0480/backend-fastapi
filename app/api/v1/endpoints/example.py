from fastapi import APIRouter
from app.schemas.common import Message

router = APIRouter()

@router.get("/hello", response_model=Message)
def hello_world():
    return {"message": "Hello from API v1"}
