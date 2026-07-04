import hashlib


def sha256_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest for a piece of text.

    Used to dedupe both whole documents and individual chunks so identical
    content is never re-embedded or re-stored.
    """
    normalized = text.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()
