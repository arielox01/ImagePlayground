from sqlalchemy import Column, String, Integer, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class Album(BaseModel):
    __tablename__ = "albums"

    title = Column(String, nullable=False)
    description = Column(Text)
    qr_code = Column(String)  # Path to stored QR code image
    access_code = Column(String, unique=True, index=True)  # Unique code for accessing the album
    is_public = Column(Boolean, default=False)

    # Foreign keys
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    owner = relationship("User", back_populates="albums")
    photos = relationship("Photo", back_populates="album", cascade="all, delete-orphan")