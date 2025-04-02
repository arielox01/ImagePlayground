from fastapi import APIRouter

from app.api.endpoints import auth, users, albums, photos

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(albums.router, prefix="/albums", tags=["albums"])
api_router.include_router(photos.router, prefix="/photos", tags=["photos"])