"""CEIO: encrypted container for pipeline images and videos (plan 10 §14, node request 06 SEC1).

Library + CLI. Needs `cryptography` >= 49 (native HPKE); run it with the ComfyUI venv python.

v1 (one recipient; still written when there is only one), big-endian:
    "CEIO" | version=1 u8 | suite u8 | flags u8 | key_id 16 | chunk_size u32     <- header (27 bytes)
    hpke_blob 80  = HPKE enc (32) || HPKE-sealed 32-byte file key (48)
    payload       = AES-256-GCM STREAM chunks of: u32 meta_len | meta JSON | body
    HPKE info = b"CEIO-v1" || SHA-256(header); chunk aad = SHA-256(header).

v2 (several recipients, e.g. a short-lived leaf plus a long-lived recovery key):
    "CEIO" | version=2 u8 | suite u8 | flags u8 | n u8 | chunk_size u32        <- fixed part (12 bytes)
    key_id 16 * n                                                               <- recipients, in order
    hpke_blob 80 * n                                                            <- the file key sealed to each
    payload       = as v1
    hh = SHA-256(fixed part || all key_ids); HPKE info = b"CEIO-v2" || hh; chunk aad = hh.
    Any one recipient's private key opens the file; the recipients' key_ids are the only cleartext.

Chunk i: nonce = i (11 bytes) || last (0x00/0x01). Nothing descriptive is outside the ciphertext.

CLI:
    ceio.py keygen --export KEYFILE          # TEST ONLY: writes a private key to disk
    ceio.py encrypt --recipient PUB [--recipient PUB2 ...] IN [--name N] [-o OUT]
    ceio.py decrypt --key KEYFILE IN [-o OUT]    # any recipient's key
    ceio.py inspect IN
"""
import argparse
import base64
import hashlib
import json
import os
import struct
import sys

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hpke, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC, VERSION, VERSION_MULTI, SUITE_X25519_AES256GCM = b"CEIO", 1, 2, 1
FLAG_INPUT, FLAG_VIDEO = 1, 2
HEADER = struct.Struct(">4sBBB16sI")          # v1: 27 bytes
HEADER_V2 = struct.Struct(">4sBBBBI")         # v2 fixed part: 12 bytes
HPKE_BLOB_LEN = 32 + 32 + 16
KEY_ID_LEN = 16
MAX_RECIPIENTS = 16
CHUNK = 64 * 1024
TAG = 16
_SUITE = hpke.Suite(hpke.KEM.X25519, hpke.KDF.HKDF_SHA256, hpke.AEAD.AES_256_GCM)


class CeioError(Exception):
    """Any format or authentication failure. Never partially trusted."""


# ------------------------------------------------------------------ keys

def generate_keypair():
    """Private key stays inside OpenSSL; only the public key is exported."""
    sk = X25519PrivateKey.generate()
    return sk, public_bytes(sk.public_key())


def public_bytes(pk):
    return pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def parse_public(b64):
    try:
        raw = base64.b64decode(b64, validate=True)
    except ValueError as e:
        raise CeioError(f"recipient is not base64: {e}") from None
    if len(raw) != 32:
        raise CeioError(f"recipient must be a 32-byte X25519 public key, got {len(raw)} bytes")
    return X25519PublicKey.from_public_bytes(raw)


def key_id_for(pub_raw):
    return hashlib.sha256(pub_raw).digest()[:16]


# ------------------------------------------------------------------ container

def _nonce(i, last):
    if i >= 1 << 88:
        raise CeioError("too many chunks")
    return i.to_bytes(11, "big") + (b"\x01" if last else b"\x00")


def _chunks(file_key, plain, hh, chunk_size):
    aead = AESGCM(file_key)
    n = max(1, -(-len(plain) // chunk_size))
    return [aead.encrypt(_nonce(i, i == n - 1), plain[i * chunk_size:(i + 1) * chunk_size], hh) for i in range(n)]


def seal(body, meta, recipient_pub, flags=0, key_id=None, chunk_size=CHUNK, extra_recipients=()):
    """Encrypt `body` (bytes) plus `meta` (dict, stays inside the ciphertext) to an X25519 public key.
    With `extra_recipients` (more X25519 public keys) the file is v2 and any of them can open it."""
    mj = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode()
    plain = struct.pack(">I", len(mj)) + mj + body
    file_key = AESGCM.generate_key(256)
    if not extra_recipients:
        pub_raw = public_bytes(recipient_pub)
        header = HEADER.pack(MAGIC, VERSION, SUITE_X25519_AES256GCM, flags, key_id or key_id_for(pub_raw),
                             chunk_size)
        hh = hashlib.sha256(header).digest()
        blob = _SUITE.encrypt(file_key, recipient_pub, info=b"CEIO-v1" + hh)
        assert len(blob) == HPKE_BLOB_LEN
        return b"".join([header, blob] + _chunks(file_key, plain, hh, chunk_size))

    pubs = [recipient_pub, *extra_recipients]
    ids = [key_id_for(public_bytes(p)) for p in pubs]
    if len(set(ids)) != len(ids):
        raise CeioError("the same key is listed twice as a recipient")
    if len(pubs) > MAX_RECIPIENTS:
        raise CeioError(f"at most {MAX_RECIPIENTS} recipients")
    header = HEADER_V2.pack(MAGIC, VERSION_MULTI, SUITE_X25519_AES256GCM, flags, len(pubs), chunk_size) + b"".join(ids)
    hh = hashlib.sha256(header).digest()
    blobs = [_SUITE.encrypt(file_key, p, info=b"CEIO-v2" + hh) for p in pubs]
    assert all(len(b) == HPKE_BLOB_LEN for b in blobs)
    return b"".join([header] + blobs + _chunks(file_key, plain, hh, chunk_size))


def read_header(data):
    """Cleartext header fields. `key_id` is the first recipient; `key_ids` lists all of them."""
    if len(data) < 6:
        raise CeioError("file too short for CEIO")
    if data[:4] != MAGIC:
        raise CeioError("not a CEIO file")
    ver = data[4]
    if ver == VERSION:
        if len(data) < HEADER.size + HPKE_BLOB_LEN + TAG:
            raise CeioError("file too short for CEIO")
        _, _, suite, flags, key_id, chunk_size = HEADER.unpack_from(data)
        key_ids, header_len, n = [key_id.hex()], HEADER.size, 1
    elif ver == VERSION_MULTI:
        if len(data) < HEADER_V2.size:
            raise CeioError("file too short for CEIO")
        _, _, suite, flags, n, chunk_size = HEADER_V2.unpack_from(data)
        if not 1 <= n <= MAX_RECIPIENTS:
            raise CeioError(f"bad recipient count {n}")
        header_len = HEADER_V2.size + KEY_ID_LEN * n
        if len(data) < header_len + HPKE_BLOB_LEN * n + TAG:
            raise CeioError("file too short for CEIO")
        key_ids = [data[HEADER_V2.size + KEY_ID_LEN * i:HEADER_V2.size + KEY_ID_LEN * (i + 1)].hex() for i in range(n)]
    else:
        raise CeioError(f"unsupported CEIO version {ver}")
    if suite != SUITE_X25519_AES256GCM:
        raise CeioError(f"unsupported CEIO suite {suite}")
    if not 1024 <= chunk_size <= 1 << 24:
        raise CeioError(f"bad chunk size {chunk_size}")
    return {"version": ver, "suite": suite, "flags": flags, "key_id": key_ids[0], "key_ids": key_ids,
            "recipients": n, "chunk_size": chunk_size, "header_len": header_len}


def open_(data, private_key):
    """-> (meta dict, body bytes). Raises CeioError on any tampering or wrong key.
    For v2, the key may be any of the recipients."""
    h = read_header(data)
    header = data[:h["header_len"]]
    hh = hashlib.sha256(header).digest()
    if h["version"] == VERSION:
        blob, info, payload_at = data[HEADER.size:HEADER.size + HPKE_BLOB_LEN], b"CEIO-v1" + hh, \
            HEADER.size + HPKE_BLOB_LEN
    else:
        mine = key_id_for(public_bytes(private_key.public_key())).hex()
        if mine not in h["key_ids"]:
            raise CeioError(f"this key (key_id {mine}) is not one of the file's recipients {h['key_ids']}")
        i = h["key_ids"].index(mine)
        at = h["header_len"] + HPKE_BLOB_LEN * i
        blob, info, payload_at = data[at:at + HPKE_BLOB_LEN], b"CEIO-v2" + hh, \
            h["header_len"] + HPKE_BLOB_LEN * h["recipients"]
    try:
        file_key = _SUITE.decrypt(blob, private_key, info=info)
    except Exception:
        raise CeioError("cannot decrypt: wrong key, or header / key block altered") from None
    aead, step = AESGCM(file_key), h["chunk_size"] + TAG
    payload = data[payload_at:]
    if not payload:
        raise CeioError("no payload")
    n = -(-len(payload) // step)
    parts = []
    for i in range(n):
        try:
            parts.append(aead.decrypt(_nonce(i, i == n - 1), payload[i * step:(i + 1) * step], hh))
        except InvalidTag:
            raise CeioError(f"chunk {i} failed authentication (tampered, reordered or truncated)") from None
    plain = b"".join(parts)
    (ml,) = struct.unpack_from(">I", plain)
    if 4 + ml > len(plain):
        raise CeioError("bad metadata length")
    return json.loads(plain[4:4 + ml]), plain[4 + ml:]


# ------------------------------------------------------------------ CLI

def _load_key(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return X25519PrivateKey.from_private_bytes(base64.b64decode(d["private"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="TEST ONLY: write a keypair to a file")
    k.add_argument("--export", required=True, help="key file to write (holds the PRIVATE key)")
    e = sub.add_parser("encrypt")
    e.add_argument("--recipient", required=True, action="append", help="base64 X25519 public key (repeatable)")
    e.add_argument("input")
    e.add_argument("--name", help="name stored inside the ciphertext (default: input file name)")
    e.add_argument("-o", "--output")
    d = sub.add_parser("decrypt")
    d.add_argument("--key", required=True)
    d.add_argument("input")
    d.add_argument("-o", "--output", help="default: the stored name next to the input")
    i = sub.add_parser("inspect")
    i.add_argument("input")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "keygen":
            sk = X25519PrivateKey.generate()
            raw = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                   serialization.NoEncryption())
            pub = base64.b64encode(public_bytes(sk.public_key())).decode()
            with open(a.export, "x", encoding="utf-8") as f:
                json.dump({"warning": "TEST KEY - private key on disk", "private": base64.b64encode(raw).decode(),
                           "public": pub}, f)
            print(pub)
        elif a.cmd == "encrypt":
            body = open(a.input, "rb").read()
            name = a.name or os.path.basename(a.input)
            out = a.output or a.input + ".ceio"
            pubs = [parse_public(r) for r in a.recipient]
            with open(out, "xb") as f:
                f.write(seal(body, {"name": name}, pubs[0], extra_recipients=pubs[1:]))
            print(out)
        elif a.cmd == "decrypt":
            meta, body = open_(open(a.input, "rb").read(), _load_key(a.key))
            out = a.output or os.path.join(os.path.dirname(a.input), os.path.basename(meta.get("name", "decrypted.bin")))
            with open(out, "xb") as f:
                f.write(body)
            print(out)
            print(json.dumps({k: v for k, v in meta.items() if k not in ("prompt", "workflow")}, ensure_ascii=False))
        elif a.cmd == "inspect":
            with open(a.input, "rb") as f:
                print(json.dumps(read_header(f.read(HEADER_V2.size + (KEY_ID_LEN + HPKE_BLOB_LEN) * MAX_RECIPIENTS + TAG))))
    except (CeioError, FileExistsError, OSError) as ex:
        print(f"ceio: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
