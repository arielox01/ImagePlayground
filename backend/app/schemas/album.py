from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class AlbumBase(BaseModel):
    title: str
    description: Optional[str] = None
    is_public: bool = False


# Properties to receive on album creation
class AlbumCreate(AlbumBase):
    pass


# Properties to receive on album update
class AlbumUpdate(AlbumBase):
    title: Optional[str] = None
    access_code: Optional[str] = None


# Properties shared by models stored in DB
class AlbumInDBBase(AlbumBase):
    id: int
    owner_id: int
    access_code: str
    qr_code: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Properties to return to client
class Album(AlbumInDBBase):
    photo_count: Optional[int] = 0


# Properties stored in DB
class AlbumInDB(AlbumInDBBase):
    pass