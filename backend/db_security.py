"""Small standard-library password hashing helpers for initial account seeding."""
from __future__ import annotations

import hashlib
import hmac
import os




def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Passwords must be at least 12 characters long.")
    salt = os.urandom(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt_hex, digest_hex = stored_hash.split("$", 2)
        if algorithm != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False
