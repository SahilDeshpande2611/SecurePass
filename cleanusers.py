from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.models import User
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clear_all_users():
    db: Session = SessionLocal()
    try:
        users_to_delete = db.query(User).all()
        if not users_to_delete:
            logger.info("No users found in the database to delete.")
        else:
            for user in users_to_delete:
                logger.info(f"Deleting user: {user.username} (role: {user.role})")
                db.delete(user)
            db.commit()
            logger.info(f"Successfully deleted {len(users_to_delete)} users from the database.")
    
    except Exception as e:
        logger.error(f"Error clearing users: {str(e)}")
        db.rollback()
    finally:
        db.close()

def add_new_users(users_to_add):
    db: Session = SessionLocal()
    try:
        for user_data in users_to_add:
            username = user_data["username"]
            password = user_data["password"]
            role = user_data["role"]

            existing_user = db.query(User).filter(User.username == username).first()
            if existing_user:
                logger.warning(f"User {username} already exists, skipping.")
                continue

            if role not in ["admin", "security", "manager"]:
                logger.error(f"Invalid role for user {username}: {role}")
                continue

            new_user = User(
                username=username,
                hashed_password=password,
                role=role
            )
            db.add(new_user)
            logger.info(f"Added new user: {username} (role: {role})")
        
        db.commit()
        logger.info(f"Successfully added {len(users_to_add)} new users.")
    
    except Exception as e:
        logger.error(f"Error adding new users: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_users()

    # new_users = [
    #     {"username": "admin_new", "password": "adminpass_new123", "role": "admin"},
    #     {"username": "security_new", "password": "securepass_new123", "role": "security"}
    # ]
    # add_new_users(new_users)