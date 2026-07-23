# load_data_to_db.py: Loads startups_clean.csv into the Supabase startups table

import os
import pandas as pd
from database import SessionLocal, Startup

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "data", "startups_clean.csv")

    df = pd.read_csv(csv_path)

    db = SessionLocal()

    try:
        existing_count = db.query(Startup).count()
        if existing_count > 0:
            print(f"Table already has {existing_count} rows. Skipping load to avoid duplicates.")
            return

        records = df.to_dict(orient="records")

        for record in records:
            startup = Startup(**record)
            db.add(startup)

        db.commit()
        print(f"Inserted {len(records)} rows into the startups table.")

    except Exception as e:
        db.rollback()
        print("Error loading data:")
        print(e)
    finally:
        db.close()


if __name__ == "__main__":
    load_data()