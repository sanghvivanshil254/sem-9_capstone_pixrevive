
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Load backend/.env
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="PixRevive MongoDB Test API")

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "pixrevive").strip()


@app.get("/")
def root():
    return {
        "message": "PixRevive MongoDB test API",
        "database_test": "/api/mongodb-test",
        "docs": "/docs",
    }


@app.get("/api/mongodb-test")
def mongodb_test():
    if not MONGODB_URI:
        raise HTTPException(
            status_code=500,
            detail="MONGODB_URI is missing from backend/.env",
        )

    client = None

    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )

        client.admin.command("ping")
        db = client[MONGODB_DATABASE]

        collection = db["_pixrevive_connection_tests"]

        document = {
            "message": "PixRevive connection test",
            "created_at": datetime.now(timezone.utc),
        }

        inserted = collection.insert_one(document)
        found = collection.find_one({"_id": inserted.inserted_id})
        collection.delete_one({"_id": inserted.inserted_id})

        return {
            "connected": True,
            "database": MONGODB_DATABASE,
            "ping": "ok",
            "write_read_delete": bool(found),
            "message": "MongoDB connection and CRUD test succeeded.",
        }

    except PyMongoError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB test failed: {exc}",
        ) from exc

    finally:
        if client is not None:
            client.close()
