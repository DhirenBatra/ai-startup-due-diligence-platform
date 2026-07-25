import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

db_url = os.getenv("DATABASE_URL")

if not db_url:
    raise ValueError("DATABASE_URL nahi mili! .env file check karo.")

try:
    engine = create_engine(db_url)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print("✅ Connection successful!")
        print("PostgreSQL version:", result.fetchone()[0])
except Exception as e:
    print("❌ Connection failed:")
    print(e)