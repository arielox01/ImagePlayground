import os
from sqlalchemy.orm import Session

from app.db.session import engine, Base
from app.models import user, album, photo
from app.core.security import get_password_hash
from app.config import settings


# Make sure all SQLAlchemy models are imported (app.models) before initializing DB
# otherwise, SQLAlchemy might fail to initialize relationships properly


def init_db(db: Session) -> None:
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Check if uploads folder exists
    if not os.path.exists(settings.UPLOAD_FOLDER):
        os.makedirs(settings.UPLOAD_FOLDER)

    # Create initial admin user if not exists
    admin_user = db.query(user.User).filter(user.User.email == "admin@example.com").first()
    if not admin_user:
        admin_user = user.User(
            email="admin@example.com",
            hashed_password=get_password_hash("adminpassword"),
            full_name="Admin User",
            is_admin=True,
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)