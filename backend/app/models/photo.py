from sqlalchemy import Column, String, Integer, ForeignKey, Float, Table, Text
from sqlalchemy.orm import relationship

from app.db.session import Base
from app.models.base import BaseModel

# Association table for faces in photos
face_appearances = Table(
    "face_appearances",
    Base.metadata,
    Column("photo_id", Integer, ForeignKey("photos.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True)
)


class Photo(BaseModel):
    __tablename__ = "photos"

    file_path = Column(String, nullable=False)
    thumbnail_path = Column(String)
    file_name = Column(String, nullable=False)
    file_size = Column(Integer)  # Size in bytes
    description = Column(Text)

    # Metadata for face recognition
    face_encodings = Column(Text)  # Stored as JSON string of face encodings

    # Foreign keys
    album_id = Column(Integer, ForeignKey("albums.id"), nullable=False)
    uploader_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    album = relationship("Album", back_populates="photos")
    uploader = relationship("User", back_populates="photos", foreign_keys=[uploader_id])

    # Many-to-many relationship for users in the photo
    people_in_photo = relationship(
        "User",
        secondary=face_appearances,
        back_populates="face_appearances"
    )