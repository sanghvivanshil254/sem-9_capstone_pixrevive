"""Create PixRevive collections/indexes and seed exactly one super-admin user.

Run from the backend directory: python init_database.py
Requires MONGODB_URI, SUPERADMIN_EMAIL, and SUPERADMIN_PASSWORD in backend/.env.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError

from database import get_database, get_client
from db_security import hash_password

load_dotenv(Path(__file__).resolve().parent / ".env")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_indexes(db) -> None:
    # USERS: one account per email, and at most one document with role=super_admin.
    db.users.create_index([("email", ASCENDING)], unique=True, name="uq_users_email")
    db.users.create_index(
        [("role", ASCENDING)],
        unique=True,
        partialFilterExpression={"role": "super_admin"},
        name="uq_single_super_admin",
    )
    db.users.create_index([("role", ASCENDING), ("created_at", DESCENDING)], name="ix_users_role_created")

    # IMAGES: image metadata and ownership; actual image bytes live on disk/object storage.
    db.images.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)], name="ix_images_owner_created")
    db.images.create_index([("user_id", ASCENDING), ("status", ASCENDING)], name="ix_images_owner_status")
    db.images.create_index([("job_id", ASCENDING)], sparse=True, name="ix_images_job")

    # RESTORATION_JOBS: each processing attempt and its state/result.
    db.restoration_jobs.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)], name="ix_jobs_owner_created")
    db.restoration_jobs.create_index([("status", ASCENDING), ("updated_at", DESCENDING)], name="ix_jobs_status_updated")

    # AUDIT_LOGS: security/admin activity trail.
    db.audit_logs.create_index([("actor_user_id", ASCENDING), ("created_at", DESCENDING)], name="ix_audit_actor_created")
    db.audit_logs.create_index([("created_at", DESCENDING)], name="ix_audit_created")

    # SYSTEM_SETTINGS: global settings, one document per key.
    db.system_settings.create_index([("key", ASCENDING)], unique=True, name="uq_settings_key")


def seed_super_admin(db) -> None:
    email = os.getenv("SUPERADMIN_EMAIL", "").strip().lower()
    password = os.getenv("SUPERADMIN_PASSWORD", "")
    name = os.getenv("SUPERADMIN_NAME", "PixRevive Super Admin").strip()

    if not email or not password:
        raise RuntimeError(
            "Set SUPERADMIN_EMAIL and SUPERADMIN_PASSWORD in backend/.env before initializing."
        )
    if len(password) < 12:
        raise RuntimeError("SUPERADMIN_PASSWORD must be at least 12 characters long.")

    existing_admin = db.users.find_one({"role": "super_admin"})
    if existing_admin:
        if existing_admin.get("email", "").lower() != email:
            raise RuntimeError(
                "A super-admin already exists with a different email. No second super-admin was created. "
                "Review the existing account manually before changing SUPERADMIN_EMAIL."
            )
        print(f"Super-admin already exists: {email} (password was not changed).")
        return

    existing_user = db.users.find_one({"email": email})
    if existing_user:
        raise RuntimeError(
            "The configured super-admin email already belongs to a non-super-admin account. "
            "No role was changed automatically; review the account manually."
        )

    now = utcnow()
    user = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "role": "super_admin",
        "is_active": True,
        "email_verified": True,
        "created_at": now,
        "updated_at": now,
        "last_login_at": None,
    }
    try:
        db.users.insert_one(user)
    except DuplicateKeyError as exc:
        raise RuntimeError("Super-admin uniqueness constraint prevented a duplicate account.") from exc
    print(f"Created the one super-admin account: {email}")


def main() -> int:
    try:
        client = get_client()
        client.admin.command("ping")
        db = get_database()
        create_indexes(db)
        seed_super_admin(db)

        # Seed only safe defaults; the super-admin can manage settings later.
        db.system_settings.update_one(
            {"key": "maintenance_mode"},
            {"$setOnInsert": {"key": "maintenance_mode", "value": False, "updated_at": utcnow()}},
            upsert=True,
        )
        db.system_settings.update_one(
            {"key": "allow_registration"},
            {"$setOnInsert": {"key": "allow_registration", "value": True, "updated_at": utcnow()}},
            upsert=True,
        )
        print(f"MongoDB initialized successfully. Database: {db.name}")
        print("Collections: users, images, restoration_jobs, audit_logs, system_settings")
        return 0
    except (RuntimeError, PyMongoError) as exc:
        print(f"[ERROR] Database initialization failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            get_client().close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
