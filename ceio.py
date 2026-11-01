"""CEIO v1: encrypted container for pipeline images and videos (plan 10 §14, node request 06 SEC1).

Library + CLI. Needs `cryptography` >= 49 (native HPKE); run it with the ComfyUI venv python.

Format (big-endian):
    "CEIO" | version u8 | suite u8 | flags u8 | key_id 16 | chunk_size u32     <- header (27 bytes)
    hpke_blob 80  = HPKE enc (32) || HPKE-sealed 32-byte file key (48)
    payload       = AES-256-GCM STREAM chunks of: u32 meta_len | meta JSON | body
HPKE suite: DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-256-GCM, Base mode, info = b"CEIO-v1" || SHA-256(header).
Chunk i: nonce = i (11 bytes) || last (0x00/0x01), aad = SHA-256(header).
Nothing descriptive is outside the ciphertext.

CLI:
    ceio.py keygen --export KEYFILE          # TEST ONLY: writes a private key to disk
    ceio.py encrypt --recipient PUB IN [--name N] [-o OUT]
    ceio.py decrypt --key KEYFILE IN [-o OUT]
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

MAGIC, VERSION, SUITE_X25519_AES256GCM = b"CEIO", 1, 1
FLAG_INPUT, FLAG_VIDEO = 1, 2
HEADER = struct.Struct(">4sBBB16sI")          # 27 bytes
HPKE_BLOB_LEN = 32 + 32 + 16
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


def seal(body, meta, recipient_pub, flags=0, key_id=None, chunk_size=CHUNK):
    """Encrypt `body` (bytes) plus `meta` (dict, stays inside the ciphertext) to an X25519 public key."""
    pub_raw = public_bytes(recipient_pub)
    header = HEADER.pack(MAGIC, VERSION, SUITE_X25519_AES256GCM, flags, key_id or key_id_for(pub_raw), chunk_size)
    hh = hashlib.sha256(header).digest()
    file_key = AESGCM.generate_key(256)
    blob = _SUITE.encrypt(file_key, recipient_pub, info=b"CEIO-v1" + hh)
    assert len(blob) == HPKE_BLOB_LEN
    mj = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode()
    plain = struct.pack(">I", len(mj)) + mj + body
    aead, out = AESGCM(file_key), [header, blob]
    n = max(1, -(-len(plain) // chunk_size))
    for i in range(n):
        out.append(aead.encrypt(_nonce(i, i == n - 1), plain[i * chunk_size:(i + 1) * chunk_size], hh))
    return b"".join(out)


def read_header(data):
    if len(data) < HEADER.size + HPKE_BLOB_LEN + TAG:
        raise CeioError("file too short for CEIO")
    magic, ver, suite, flags, key_id, chunk_size = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise CeioError("not a CEIO file")
    if ver != VERSION or suite != SUITE_X25519_AES256GCM:
        raise CeioError(f"unsupported CEIO version {ver} / suite {suite}")
    if not 1024 <= chunk_size <= 1 << 24:
        raise CeioError(f"bad chunk size {chunk_size}")
    return {"version": ver, "suite": suite, "flags": flags, "key_id": key_id.hex(), "chunk_size": chunk_size}


def open_(data, private_key):
    """-> (meta dict, body bytes). Raises CeioError on any tampering or wrong key."""
    h = read_header(data)
    header = data[:HEADER.size]
    hh = hashlib.sha256(header).digest()
    blob = data[HEADER.size:HEADER.size + HPKE_BLOB_LEN]
    try:
        file_key = _SUITE.decrypt(blob, private_key, info=b"CEIO-v1" + hh)
    except Exception:
        raise CeioError("cannot decrypt: wrong key, or header / key block altered") from None
    aead, step = AESGCM(file_key), h["chunk_size"] + TAG
    payload = data[HEADER.size + HPKE_BLOB_LEN:]
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
    d = json.load(open(path, encoding="utf-8"))
    return X25519PrivateKey.from_private_bytes(base64.b64decode(d["private"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="TEST ONLY: write a keypair to a file")
    k.add_argument("--export", required=True, help="key file to write (holds the PRIVATE key)")
    e = sub.add_parser("encrypt")
    e.add_argument("--recipient", required=True, help="base64 X25519 public key")
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
            with open(out, "xb") as f:
                f.write(seal(body, {"name": name}, parse_public(a.recipient)))
            print(out)
        elif a.cmd == "decrypt":
            meta, body = open_(open(a.input, "rb").read(), _load_key(a.key))
            out = a.output or os.path.join(os.path.dirname(a.input), os.path.basename(meta.get("name", "decrypted.bin")))
            with open(out, "xb") as f:
                f.write(body)
            print(out)
            print(json.dumps({k: v for k, v in meta.items() if k not in ("prompt", "workflow")}, ensure_ascii=False))
        elif a.cmd == "inspect":
            print(json.dumps(read_header(open(a.input, "rb").read(HEADER.size + HPKE_BLOB_LEN + TAG))))
    except (CeioError, FileExistsError, OSError) as ex:
        print(f"ceio: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
