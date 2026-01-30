from fastapi import APIRouter
from app.api.v1.endpoints import example, chat, travel, demo, recommend

api_router = APIRouter()
api_router.include_router(example.router, prefix="/example", tags=["example"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(travel.router, prefix="/travel", tags=["travel"])
api_router.include_router(demo.router, prefix="/demo", tags=["demo"])
api_router.include_router(recommend.router, prefix="/recommend", tags=["recommend"])
