import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# load .env file (make sure you have DATABASE_URL set)
load_dotenv()

# Ensure the db directory exists (for volume-mounted SQLite persistence)
os.makedirs("db", exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/openmemory.db")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment")

# SQLAlchemy engine & session
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
