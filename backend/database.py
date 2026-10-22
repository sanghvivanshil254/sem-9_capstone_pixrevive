
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, AsyncMongoClient

# Load environment variables from backend/.env
load_dotenv(Path(__file__).resolve().parent / ".env")

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017"
).strip()

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "pixrevive"
).strip()

# Async connection for the FastAPI application
client = AsyncMongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]

# Separate synchronous connection for init_database.py
_sync_client = None


async def connect_db():
    """Connect to MongoDB and verify the connection."""
    await client.admin.command("ping")
    print("MongoDB connected successfully!")


async def close_db():
    """Close the asynchronous MongoDB connection."""
    await client.close()


def get_client():
    """Return a synchronous client for database initialization."""
    global _sync_client

    if _sync_client is None:
        _sync_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )

    return _sync_client


def get_database():
    """Return the synchronous database for init_database.py."""
    return get_client()[MONGODB_DATABASE]


def close_sync_database():
    """Close the synchronous initialization connection."""
    global _sync_client

    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None
