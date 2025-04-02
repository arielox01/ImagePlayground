import os
import numpy as np
import face_recognition
from PIL import Image
import json
from typing import List, Tuple, Optional, Dict, Any


def process_image_for_faces(
        image_path: str,
        photo_id: int,
        thumbnail_dir: str
) -> Tuple[List[List[float]], Optional[str]]:
    """
    Process an image for face detection and generate a thumbnail.

    Args:
        image_path: Path to the image file
        photo_id: ID of the photo
        thumbnail_dir: Directory to save the thumbnail

    Returns:
        Tuple of (face_encodings, thumbnail_path)
    """
    try:
        # Create thumbnail directory if it doesn't exist
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = os.path.join(thumbnail_dir, f"thumb_{photo_id}.jpg")

        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image)

        # Create face encodings
        face_encodings = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(image, face_locations)
            # Convert numpy arrays to lists for JSON serialization
            face_encodings = [encoding.tolist() for encoding in face_encodings]

        # Create a thumbnail
        pil_image = Image.open(image_path)
        pil_image.thumbnail((300, 300))
        pil_image.save(thumbnail_path, "JPEG")

        return face_encodings, thumbnail_path

    except Exception as e:
        print(f"Error processing image for faces: {e}")
        return [], None


def match_face_to_users(
        db_session,
        face_encoding: List[float],
        tolerance: float = 0.6
) -> Optional[int]:
    """
    Match a face encoding to a user in the database.

    Args:
        db_session: Database session
        face_encoding: Face encoding to match
        tolerance: Tolerance level for face comparison (lower is stricter)

    Returns:
        User ID if match found, None otherwise
    """
    from app.models.photo import Photo
    from app.models.user import User

    # Convert the face encoding to a numpy array
    face_to_match = np.array(face_encoding)

    # Get all photos with face encodings
    photos_with_faces = db_session.query(Photo).filter(Photo.face_encodings.isnot(None)).all()

    # Track faces that might be the same person
    potential_matches = {}

    for photo in photos_with_faces:
        if not photo.face_encodings:
            continue

        photo_face_encodings = json.loads(photo.face_encodings)
        if not photo_face_encodings:
            continue

        # Check each face in the photo
        for idx, encoding in enumerate(photo_face_encodings):
            # Check if the faces match
            face_distance = face_recognition.face_distance([np.array(encoding)], face_to_match)[0]
            if face_distance <= tolerance:
                # Get users associated with this photo
                for user in photo.people_in_photo:
                    if user.id not in potential_matches:
                        potential_matches[user.id] = 0
                    potential_matches[user.id] += 1

    # Return the user ID with the most matches, if any
    if potential_matches:
        return max(potential_matches.items(), key=lambda x: x[1])[0]

    return None


def tag_faces_in_photo(db_session, photo_id: int) -> None:
    """
    Identify and tag faces in a photo.

    Args:
        db_session: Database session
        photo_id: ID of the photo to process
    """
    from app.models.photo import Photo
    from app.models.user import User

    # Get the photo
    photo = db_session.query(Photo