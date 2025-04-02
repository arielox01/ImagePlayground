from typing import Optional, List, Any
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class PhotoBase(BaseModel):
    description: Optional[str] = None


# Properties to receive on photo creation
class PhotoCreate(PhotoBase):
    album_id: int


# Properties to receive on photo update
class PhotoUpdate(PhotoBase):
    pass


# Properties shared by models stored in DB
class PhotoInDBBase(PhotoBase):
    id: int
    file_path: str
    thumbnail_path: Optional[str] = None
    file_name: str
    file_size: Optional[int] = None
    album_id: int
    uploader_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Properties to return to client
class Photo(PhotoInDBBase):
    people_in_photo: Optional[List[int]] = []  # List of user IDs


# Properties stored in DB
class PhotoInDB(PhotoInDBBase):
    face_encodings: Optional[str] = None