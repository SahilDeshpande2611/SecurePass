from app.core.database import user_engine, photo_engine, Base

def init_db():
    print("Creating tables for user_management.db")
    Base.metadata.create_all(bind=user_engine)
    print("Creating tables for captured_images.db")
    Base.metadata.create_all(bind=photo_engine)
    print("Database initialization complete")

if __name__ == "__main__":
    init_db()