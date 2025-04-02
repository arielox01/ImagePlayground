from sqlalchemy import Boolean, Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class User(BaseModel):
    __tablename__ = "users"

    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    # Relationships
    albums = relationship("Album", back_populates="owner")
    photos = relationship("Photo", back_populates="uploader")

    # Many-to-many relationship for users appearing in photos (faces)
    face_appearances = relationship(
        "Photo",
        secondary="face_appearances",
        back_populates="people_in_photo"
    )