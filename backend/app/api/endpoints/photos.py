import os
import shutil
import uuid
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import json

from app import models, schemas
from app.api import deps
from app.config import settings
from app.core.face_recognition import process_image_for_faces

router = APIRouter()


def create_upload_file(file: UploadFile, subdirectory: str) -> tuple:
    """Upload a file to the specified subdirectory and return file info"""
    upload_dir = os.path.join(settings.UPLOAD_FOLDER, subdirectory)
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(upload_dir, filename)
    rel_path = os.path.join(subdirectory, filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get file size
    file_size = os.path.getsize(file_path)

    return rel_path, filename, file_size


@router.get("/", response_model=List[schemas.Photo])
def read_photos(
        db: Session = Depends(deps.get_db),
        skip: int = 0,
        limit: int = 100,
        album_id: int = None,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve photos.
    """
    query = db.query(models.Photo)

    # Filter by album if specified
    if album_id is not None:
        album = db.query(models.Album).filter(models.Album.id == album_id).first()
        if not album:
            raise HTTPException(status_code=404, detail="Album not found")

        # Check permissions for private albums
        if not album.is_public and album.owner_id != current_user.id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        query = query.filter(models.Photo.album_id == album_id)
    else:
        # Get photos from albums the user has access to
        if not current_user.is_admin:
            query = query.join(models.Album).filter(
                (models.Album.owner_id == current_user.id) |
                (models.Album.is_public == True)
            )

    photos = query.offset(skip).limit(limit).all()

    # Add people in photo information
    for photo in photos:
        photo.people_in_photo = [user.id for user in photo.people_in_photo]

    return photos


@router.post("/", response_model=schemas.Photo)
async def upload_photo(
        db: Session = Depends(deps.get_db),
        file: UploadFile = File(...),
        album_id: int = Form(...),
        description: str = Form(None),
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Upload a new photo.
    """
    # Check if album exists and user has access
    album = db.query(models.Album).filter(models.Album.id == album_id).first()
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    # Check content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Upload file
    file_path, file_name, file_size = create_upload_file(file, f"photos/album_{album_id}")

    # Create photo record
    photo = models.Photo(
        file_path=file_path,
        file_name=file_name,
        file_size=file_size,
        description=description,
        album_id=album_id,
        uploader_id=current_user.id,
    )

    db.add(photo)
    db.commit()
    db.refresh(photo)

    # Process image for face recognition (this will be implemented in the face_recognition.py)
    full_path = os.path.join(settings.UPLOAD_FOLDER, file_path)
    face_encodings, thumbnail_path = process_image_for_faces(
        full_path,
        photo.id,
        os.path.join(settings.UPLOAD_FOLDER, f"thumbnails/album_{album_id}")
    )

    # Update photo with face encodings and thumbnail path
    if thumbnail_path:
        photo.thumbnail_path = os.path.relpath(
            thumbnail_path, settings.UPLOAD_FOLDER
        )

    if face_encodings:
        photo.face_encodings = json.dumps(face_encodings)

    db.add(photo)
    db.commit()
    db.refresh(photo)

    return photo


@router.get("/{photo_id}", response_model=schemas.Photo)
def read_photo(
        *,
        db: Session = Depends(deps.get_db),
        photo_id: int,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get photo by ID.
    """
    photo = db.query(models.Photo).filter(models.Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    # Check if user has access to the album
    album = db.query(models.Album).filter(models.Album.id == photo.album_id).first()
    if not album.is_public and album.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Add people in photo information
    photo.people_in_photo = [user.id for user in photo.people_in_photo]

    return photo


@router.put("/{photo_id}", response_model=schemas.Photo)
def update_photo(
        *,
        db: Session = Depends(deps.get_db),
        photo_id: int,
        photo_in: schemas.PhotoUpdate,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Update photo metadata.
    """
    photo = db.query(models.Photo).filter(models.Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    # Check if user has permission
    if photo.uploader_id != current_user.id and not current_user.is_admin:
        album = db.query(models.Album).filter(models.Album.id == photo.album_id).first()
        if album.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")

    # Update photo
    photo_data = photo_in.dict(exclude_unset=True)
    for field, value in photo_data.items():
        setattr(photo, field, value)

    db.add(photo)
    db.commit()
    db.refresh(photo)

    # Add people in photo information
    photo.people_in_photo = [user.id for user in photo.people_in_photo]

    return photo


@router.delete("/{photo_id}", response_model=schemas.Photo)
def delete_photo(
        *,
        db: Session = Depends(deps.get_db),
        photo_id: int,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Delete photo.
    """
    photo = db.query(models.Photo).filter(models.Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    # Check if user has permission
    if photo.uploader_id != current_user.id and not current_user.is_admin:
        album = db.query(models.Album).filter(models.Album.id == photo.album_id).first()
        if album.owner_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")

    # Delete photo files
    if photo.file_path and os.path.exists(os.path.join(settings.UPLOAD_FOLDER, photo.file_path)):
        os.remove(os.path.join(settings.UPLOAD_FOLDER, photo.file_path))
    if photo.thumbnail_path and os.path.exists(os.path.join(settings.UPLOAD_FOLDER, photo.thumbnail_path)):
        os.remove(os.path.join(settings.UPLOAD_FOLDER, photo.thumbnail_path))

    # Keep a copy of the photo to return
    photo_copy = schemas.Photo.from_orm(photo)

    db.delete(photo)
    db.commit()

    return photo_copy


@router.get("/download/{photo_id}")
def download_photo(
        *,
        db: Session = Depends(deps.get_db),
        photo_id: int,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Download photo.
    """
    photo = db.query(models.Photo).filter(models.Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    # Check if user has access to the album
    album = db.query(models.Album).filter(models.Album.id == photo.album_id).first()
    if not album.is_public and album.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    file_path = os.path.join(settings.UPLOAD_FOLDER, photo.file_path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=photo.file_name
    )


@router.get("/by-person/{user_id}", response_model=List[schemas.Photo])
def get_photos_by_person(
        *,
        db: Session = Depends(deps.get_db),
        user_id: int,
        album_id: int = None,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get photos containing a specific person.
    """
    # Check if person exists
    person = db.query(models.User).filter(models.User.id == user_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Get photos with this person
    query = db.query(models.Photo).filter(
        models.Photo.people_in_photo.any(id=user_id)
    )

    # Filter by album if specified
    if album_id is not None:
        album = db.query(models.Album).filter(models.Album.id == album_id).first()
        if not album:
            raise HTTPException(status_code=404, detail="Album not found")

        # Check permissions for private albums
        if not album.is_public and album.owner_id != current_user.id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Not enough permissions")

        query = query.filter(models.Photo.album_id == album_id)
    else:
        # Get photos from albums the user has access to
        if not current_user.is_admin:
            query = query.join(models.Album).filter(
                (models.Album.owner_id == current_user.id) |
                (models.Album.is_public == True)
            )

    photos = query.all()

    # Add people in photo information
    for photo in photos:
        photo.people_in_photo = [user.id for user in photo.people_in_photo]

    return photos