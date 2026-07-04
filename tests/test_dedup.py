from app.dedup import sha256_hash


def test_same_content_same_hash():
    a = sha256_hash("Revenue grew 12% year over year.")
    b = sha256_hash("Revenue grew 12% year over year.")
    assert a == b


def test_whitespace_normalized():
    a = sha256_hash("  Revenue grew 12%.  ")
    b = sha256_hash("Revenue grew 12%.")
    assert a == b


def test_different_content_different_hash():
    a = sha256_hash("Revenue grew 12%.")
    b = sha256_hash("Revenue declined 5%.")
    assert a != b


def test_hash_is_hex_sha256_length():
    h = sha256_hash("some financial text")
    assert len(h) == 64
    int(h, 16)  # raises if not valid hex
