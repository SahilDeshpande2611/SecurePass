from app.core.database import Base, engine
from app.core.models import AuthorizedVehicle, AccessLog

# Drop existing tables
Base.metadata.drop_all(bind=engine)
# Recreate tables
Base.metadata.create_all(bind=engine)
print("Database schema CLEARED successfully.")
