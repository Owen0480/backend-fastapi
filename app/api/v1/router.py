from fastapi import APIRouter
from app.api.v1.endpoints import example, chat

api_router = APIRouter()
api_router.include_router(example.router, prefix="/example", tags=["example"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
