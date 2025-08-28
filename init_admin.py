from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.models import User
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_initial_admin():
    db: Session = SessionLocal()
    try:
        # Check if admin user already exists
        existing_admin = db.query(User).filter(User.username == "ADMIN").first()
        if existing_admin:
            logger.info("Admin user 'ADMIN' already exists. Skipping creation.")
            return

        # Create admin user
        admin_user = User(
            username="ADMIN",
            hashed_password="ADMIN",  # In production, this should be hashed
            role="admin"
        )
        db.add(admin_user)
        db.commit()
        logger.info("Initial admin user 'ADMIN' created successfully.")
    except Exception as e:
        logger.error(f"Error creating initial admin user: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    add_initial_admin()