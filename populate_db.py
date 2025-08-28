import csv
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.database import UserSessionLocal
from app.core.models import AuthorizedVehicle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_vehicles():
    db: Session = UserSessionLocal()
    try:
        logger.info("Starting to populate vehicles from vehicles.csv")

        # Clear the authorized_vehicles table
        logger.info("Clearing existing vehicles from the database")
        db.query(AuthorizedVehicle).delete()
        db.commit()

        # Read vehicles from CSV
        with open("vehicles.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            vehicles = []
            for row in reader:
                # Try parsing added_at with microseconds, fall back to without
                try:
                    added_at = datetime.strptime(row['added_at'], '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    added_at = datetime.strptime(row['added_at'], '%Y-%m-%d %H:%M:%S')
                
                vehicle = AuthorizedVehicle(
                    plate_number=row['plate_number'],
                    owner_name=row['owner_name'],
                    vehicle_type=row['vehicle_type'],
                    added_at=added_at
                )
                vehicles.append(vehicle)

        # Insert vehicles one by one to handle duplicates gracefully
        inserted_count = 0
        for vehicle in vehicles:
            # Check if the plate_number already exists
            existing = db.query(AuthorizedVehicle).filter(AuthorizedVehicle.plate_number == vehicle.plate_number).first()
            if existing:
                logger.warning(f"Skipping duplicate plate_number: {vehicle.plate_number}")
                continue
            db.add(vehicle)
            inserted_count += 1

        db.commit()
        logger.info(f"Successfully populated {inserted_count} vehicles (out of {len(vehicles)} total)")
    except Exception as e:
        logger.error(f"Error populating vehicles: {str(e)}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    populate_vehicles()