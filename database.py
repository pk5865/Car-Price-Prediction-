import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# FORCE SQLite - Simple and always works
DATABASE_URL = "sqlite:///./car_prices.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ==================== MODELS ====================

class Car(Base):
    __tablename__ = "cars"
    id = Column(Integer, primary_key=True, index=True)
    company = Column(String(100), index=True)
    model = Column(String(100), index=True)
    year = Column(Integer, index=True)
    price = Column(Float)
    fuel_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class CarListing(Base):
    __tablename__ = "car_listings"
    id = Column(Integer, primary_key=True, index=True)
    company = Column(String(100))
    model = Column(String(100))
    year = Column(Integer)
    fuel_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class SearchHistory(Base):
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, index=True)
    company = Column(String(100))
    model = Column(String(100))
    year = Column(Integer)
    fuel_type = Column(String(50))
    kms_driven = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    predicted_low = Column(Float, nullable=True)
    predicted_high = Column(Float, nullable=True)
    onroad_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==================== FUNCTIONS ====================

def init_db():
    """Initialize database."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ SQLite Database Ready!")
    except Exception as e:
        print(f"❌ Database error: {e}")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
