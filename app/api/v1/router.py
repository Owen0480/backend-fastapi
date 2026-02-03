from fastapi import APIRouter
from app.api.v1.endpoints import travel, demo, travel_style

api_router = APIRouter()
api_router.include_router(travel.router, prefix="/travel", tags=["travel"])
api_router.include_router(demo.router, prefix="/demo", tags=["demo"])
api_router.include_router(travel_style.router, prefix="/travel-style", tags=["travel-style"])