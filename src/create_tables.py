# create_tables.py: Creates all tables defined in database.py

from database import Base, engine

if __name__ == "__main__":
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully: startups, scores, reports")