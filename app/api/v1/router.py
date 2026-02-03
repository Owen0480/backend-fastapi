from fastapi import APIRouter
<<<<<<< HEAD
from app.api.v1.endpoints import example, chat, travel, demo, recommend
=======
from app.api.v1.endpoints import travel, demo, travel_style
>>>>>>> mei

api_router = APIRouter()
api_router.include_router(travel.router, prefix="/travel", tags=["travel"])
api_router.include_router(demo.router, prefix="/demo", tags=["demo"])
<<<<<<< HEAD
api_router.include_router(recommend.router, prefix="/recommend", tags=["recommend"])
=======
api_router.include_router(travel_style.router, prefix="/travel-style", tags=["travel-style"])
>>>>>>> mei
