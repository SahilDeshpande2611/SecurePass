from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.models import User
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create engine and session for user_management.db
engine = create_engine("sqlite:///C:/Users/PRATIK/SECURE_PASS/SECURE_PASS_BACKEND/user_management.db", echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def add_users():
    # Create a new session
    db = SessionLocal()
    try:
        # Check if MANAGER already exists
        manager_user = db.query(User).filter(User.username == "MANAGER").first()
        if not manager_user:
            manager = User(
                username="MANAGER",
                hashed_password="MANAGER",
                role="manager"
            )
            db.add(manager)
            logger.info("Added user MANAGER with role manager")
        else:
            logger.info("User MANAGER already exists, skipping")

        # Check if SECURITY already exists
        security_user = db.query(User).filter(User.username == "SECURITY").first()
        if not security_user:
            security = User(
                username="SECURITY",
                hashed_password="SECURITY",
                role="security"
            )
            db.add(security)
            logger.info("Added user SECURITY with role security")
        else:
            logger.info("User SECURITY already exists, skipping")

        # Commit the changes
        db.commit()
    except Exception as e:
        logger.error(f"Error adding users: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    add_users()