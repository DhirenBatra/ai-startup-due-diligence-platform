# database.py: Database connection setup and ORM models

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, BigInteger, Float, String, ForeignKey, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Engine: manages the actual connection pool to the database
engine = create_engine(DATABASE_URL)

# Session factory: creates a new session per request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that all ORM models inherit from
Base = declarative_base()


class Startup(Base):
    __tablename__ = "startups"

    id = Column(Integer, primary_key=True, index=True)
    state_code = Column(String)
    labels = Column(Integer)  # actual outcome: 1 = success, 0 = failure
    age_first_funding_year = Column(Float)
    age_last_funding_year = Column(Float)
    age_first_milestone_year = Column(Float)
    age_last_milestone_year = Column(Float)
    relationships = Column(Integer)
    funding_rounds = Column(Integer)
    funding_total_usd = Column(BigInteger)
    milestones = Column(Integer)
    is_CA = Column(Integer)
    is_NY = Column(Integer)
    is_MA = Column(Integer)
    is_TX = Column(Integer)
    is_otherstate = Column(Integer)
    is_software = Column(Integer)
    is_web = Column(Integer)
    is_mobile = Column(Integer)
    is_enterprise = Column(Integer)
    is_advertising = Column(Integer)
    is_gamesvideo = Column(Integer)
    is_ecommerce = Column(Integer)
    is_biotech = Column(Integer)
    is_consulting = Column(Integer)
    is_othercategory = Column(Integer)
    has_VC = Column(Integer)
    has_angel = Column(Integer)
    has_roundA = Column(Integer)
    has_roundB = Column(Integer)
    has_roundC = Column(Integer)
    has_roundD = Column(Integer)
    avg_participants = Column(Float)
    is_top500 = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    scores = relationship("Score", back_populates="startup")
    reports = relationship("Report", back_populates="startup")


class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.id"))
    success_probability = Column(Float)
    top_factors = Column(Text)  # stored as a JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    startup = relationship("Startup", back_populates="scores")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    startup_id = Column(Integer, ForeignKey("startups.id"))
    report_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    startup = relationship("Startup", back_populates="reports")


def get_db():
    """Dependency used by FastAPI endpoints to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()