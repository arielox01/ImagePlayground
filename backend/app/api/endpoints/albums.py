import os
import secrets
import qrcode
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import models, schemas
from app.api import deps
from app.config import settings

router = APIRouter()


def generate_access_code():
    """Generate a random access code for albums"""
    return secrets.token_urlsafe(8)


def generate_qr_code(access_code: str, album_id: int) -> str:
    """Generate QR code image and return path"""
    qr_dir = os.path.join(settings.UPLOAD_FOLDER, "qrcodes")
    os.makedirs(qr_dir, exist_ok=True)

    qr_path = f"qrcodes/album_{album_id}_{access_code}.png"
    full_path = os.path.join(settings.UPLOAD_FOLDER, qr_path)

    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(f"album/{access_code}")
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(full_path)

    return qr_path


@router.get("/", response_model=List[schemas.Album])
def read_albums(
        db: Session = Depends(deps.get_db),
        skip: int = 0,
        limit: int = 100,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve albums.
    """
    if current_user.is_admin:
        albums = db.query(models.Album).offset(skip).limit(limit).all()
    else:
        albums = (
            db.query(models.Album)
                .filter(
                (models.Album.owner_id == current_user.id) |
                (models.Album.is_public == True)
            )
                .offset(skip)
                .limit(limit)
                .all()
        )

    # Add photo count
    for album in albums:
        album.photo_count = db.query(models.Photo).filter(models.Photo.album_id == album.id).count()

    return albums


@router.post("/", response_model=schemas.Album)
def create_album(
        *,
        db: Session = Depends(deps.get_db),
        album_in: schemas.AlbumCreate,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Create new album.
    """
    # Only admins can create albums
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Not enough permissions to create albums"
        )

    # Generate unique access code
    access_code = generate_access_code()

    # Create album
    album = models.Album(
        **album_in.dict(),
        owner_id=current_user.id,
        access_code=access_code
    )

    db.add(album)
    db.commit()
    db.refresh(album)

    # Generate QR code
    qr_path = generate_qr_code(access_code, album.id)
    album.qr_code = qr_path

    db.add(album)
    db.commit()
    db.refresh(album)

    return album


@router.get("/{album_id}", response_model=schemas.Album)
def read_album(
        *,
        db: Session = Depends(deps.get_db),
        album_id: int,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get album by ID.
    """
    album = db.query(models.Album).filter(models.Album.id == album_id).first()
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    # Check permissions
    if not album.is_public and album.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Add photo count
    album.photo_count = db.query(models.Photo).filter(models.Photo.album_id == album.id).count()

    return album


@router.get("/by-code/{access_code}", response_model=schemas.Album)
def read_album_by_code(
        *,
        db: Session = Depends(deps.get_db),
        access_code: str,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get album by access code.
    """
    album = db.query(models.Album).filter(models.Album.access_code == access_code).first()
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    # Add photo count
    album.photo_count = db.query(models.Photo).filter(models.Photo.album_id == album.id).count()

    return album


@router.put("/{album_id}", response_model=schemas.Album)
def update_album(
        *,
        db: Session = Depends(deps.get_db),
        album_id: int,
        album_in: schemas.AlbumUpdate,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Update an album.
    """
    album = db.query(models.Album).filter(models.Album.id == album_id).first()
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    # Check permissions
    if album.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Update album
    album_data = album_in.dict(exclude_unset=True)
    for field, value in album_data.items():
        setattr(album, field, value)

    # If access code is updated, regenerate QR code
    if "access_code" in album_data:
        qr_path = generate_qr_code(album.access_code, album.id)
        album.qr_code = qr_path

    db.add(album)
    db.commit()
    db.refresh(album)

    # Add photo count
    album.photo_count = db.query(models.Photo).filter(models.Photo.album_id == album.id).count()

    return album


@router.delete("/{album_id}", response_model=schemas.Album)
def delete_album(
        *,
        db: Session = Depends(deps.get_db),
        album_id: int,
        current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Delete an album.
    """
    album = db.query(models.Album).filter(models.Album.id == album_id).first()
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    # Check permissions
    if album.owner_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    # Delete associated photos
    photos = db.query(models.Photo).filter(models.Photo.album_id == album_id).all()
    for photo in photos:
        # Delete photo files
        if photo.file_path and os.path.exists(os.path.join(settings.UPLOAD_FOLDER, photo.file_path)):
            os.remove(os.path.join(settings.UPLOAD_FOLDER, photo.file_path))
        if photo.thumbnail_path and os.path.exists(os.path.join(settings.UPLOAD_FOLDER, photo.thumbnail_path)):
            os.remove(os.path.join(settings.UPLOAD_FOLDER, photo.thumbnail_path))

        db.delete(photo)

    # Delete QR code
    if album.qr_code and os.path.exists(os.path.join(settings.UPLOAD_FOLDER, album.qr_code)):
        os.remove(os.path.join(settings.UPLOAD_FOLDER, album.qr_code))

    db.delete(album)
    db.commit()

    return album