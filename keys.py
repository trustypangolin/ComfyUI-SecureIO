"""
TEST ONLY key nodes, signed by the local test root CA (G:\\Ai\\build\\ca\\test-root, override SECUREIO_TEST_ROOT).

- SecureIOIssueTestLeaf: a fresh X25519 leaf per run, private key into the key dir (default R:\\leaf).
- SecureIOTestRecoveryKey: the long-lived X25519 recovery key, created once next to the test root
  (recovery.ceio.json + recovery.crt) and reused. Wire its recipient to a saver's recovery_recipient and any
  sealed file can be opened with it after the leaf is gone.

Both read the TEST root key inside ComfyUI (and the recovery node keeps the recovery private key on the host
disk), so every custom node in the process could read them: acceptable for test keys, never for real ones.
Where the real root and recovery keys live is still to be decided.
"""

import base64
import datetime as dt
import json
import os
import re

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.x509.oid import NameOID

from . import ceio
from .loaders import key_dir, recovery_key_path

CATEGORY = "SecureIO"
UTC = dt.timezone.utc
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# X.509 needs a validity period, but nothing checks it: access ends when the leaf key is destroyed.
LEAF_CERT_VALIDITY = dt.timedelta(days=7)


def test_root_dir():
    return os.path.abspath(os.environ.get("SECUREIO_TEST_ROOT", r"G:\Ai\build\ca\test-root"))


def _load_root_cert():
    with open(os.path.join(test_root_dir(), "root.crt"), "rb") as f:
        return x509.load_pem_x509_certificate(f.read())


def _load_root():
    with open(os.path.join(test_root_dir(), "root.key"), "rb") as f:
        key = serialization.load_pem_private_key(f.read(), None)
    return key, _load_root_cert()


def _x25519_cert(root_key, root_cert, cn, sk, lifetime):
    now = dt.datetime.now(UTC)
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CEIO test CA"),
                                     x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
            .issuer_name(root_cert.subject).public_key(sk.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + lifetime)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=True, key_cert_sign=False,
                                         crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
            .sign(root_key, hashes.SHA256()))
    cert.verify_directly_issued_by(root_cert)   # chain check before anything is handed out
    return cert


def _write_key(key_path, crt_path, sk, cert, warning):
    raw = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    recipient = base64.b64encode(ceio.public_bytes(sk.public_key())).decode()
    with open(key_path, "x", encoding="utf-8") as f:
        json.dump({"warning": warning, "private": base64.b64encode(raw).decode(), "public": recipient}, f)
    with open(crt_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return recipient


def _summary(title, cert, root_cert, recipient, key_path, crt_path):
    return "\n".join([
        title,
        f"subject   : {cert.subject.rfc4514_string()}",
        f"issuer    : {cert.issuer.rfc4514_string()}",
        f"root sha256: {root_cert.fingerprint(hashes.SHA256()).hex()}",
        f"cert dates: {cert.not_valid_before_utc:%Y-%m-%d %H:%M}Z -> {cert.not_valid_after_utc:%Y-%m-%d %H:%M}Z "
        "(not enforced when opening)",
        "usage     : keyAgreement only, CA:FALSE",
        f"recipient : {recipient}",
        f"key_id    : {ceio.key_id_for(base64.b64decode(recipient)).hex()}",
        f"private   : {key_path}",
        f"cert      : {crt_path}",
    ])


class SecureIOIssueTestLeaf:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "e2e", "tooltip": "Leaf name; files are <name>_<UTC time>.* in the key dir."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("key_file", "cert_path", "recipient", "key_id", "info")
    FUNCTION = "issue"
    CATEGORY = CATEGORY
    DESCRIPTION = ("TEST ONLY. Issues a new X25519 leaf signed by the local test root into R:\\leaf "
                   "(private key on the RAM disk). Wire recipient to an encrypted saver and key_file to the loader.")

    @classmethod
    def IS_CHANGED(cls, name):
        return float("nan")   # a fresh leaf on every run

    def issue(self, name):
        if not _NAME_RE.match(name or ""):
            raise ValueError("name must be 1-40 letters, digits, '_' or '-'")
        kdir = key_dir()
        if not os.path.isdir(os.path.splitdrive(kdir)[0] + "\\"):
            raise ValueError(f"key dir drive for {kdir} is not mounted (RAM disk R:?)")
        os.makedirs(kdir, exist_ok=True)
        root_key, root_cert = _load_root()
        base = f"{name}_{dt.datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        sk = X25519PrivateKey.generate()
        cert = _x25519_cert(root_key, root_cert, base, sk, LEAF_CERT_VALIDITY)
        key_file, cert_path = f"{base}.ceio.json", os.path.join(kdir, f"{base}.crt")
        recipient = _write_key(os.path.join(kdir, key_file), cert_path, sk, cert, "TEST LEAF KEY - RAM disk only")
        info = _summary("TEST leaf issued (chain verified)", cert, root_cert, recipient,
                        os.path.join(kdir, key_file), cert_path)
        return (key_file, cert_path, recipient, ceio.key_id_for(base64.b64decode(recipient)).hex(), info)


class SecureIOTestRecoveryKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("recipient", "key_id", "info")
    FUNCTION = "get"
    CATEGORY = CATEGORY
    DESCRIPTION = ("TEST ONLY. The long-lived recovery key next to the test root (created on first use). Wire "
                   "recipient to a saver's recovery_recipient; the loader's '(recovery key)' opens such files.")

    @classmethod
    def IS_CHANGED(cls):
        p = recovery_key_path()[:-len(".ceio.json")] + ".crt"
        return os.path.getmtime(p) if os.path.exists(p) else float("nan")

    def get(self):
        key_path = recovery_key_path()
        crt_path = key_path[:-len(".ceio.json")] + ".crt"
        if os.path.exists(crt_path):
            # Normal case: public material only. Neither the recovery private key nor the root key is read.
            title = "TEST recovery key (existing; read from its cert, private key not touched)"
            root_cert = _load_root_cert()
            with open(crt_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            cert.verify_directly_issued_by(root_cert)
            recipient = base64.b64encode(ceio.public_bytes(cert.public_key())).decode()
        elif os.path.exists(key_path):
            raise ValueError(f"{key_path} exists but its cert {crt_path} is missing; restore the cert first")
        else:
            title = "TEST recovery key (created now)"
            root_key, root_cert = _load_root()   # only on creation: the cert has to be signed
            sk = X25519PrivateKey.generate()
            cert = _x25519_cert(root_key, root_cert, "CEIO test recovery", sk, dt.timedelta(days=3650))
            recipient = _write_key(key_path, crt_path, sk, cert, "TEST RECOVERY KEY - host disk, test only")
        info = _summary(title, cert, root_cert, recipient, key_path, crt_path)
        return (recipient, ceio.key_id_for(base64.b64decode(recipient)).hex(), info)


NODE_CLASS_MAPPINGS = {"SecureIOIssueTestLeaf": SecureIOIssueTestLeaf,
                       "SecureIOTestRecoveryKey": SecureIOTestRecoveryKey}
NODE_DISPLAY_NAME_MAPPINGS = {"SecureIOIssueTestLeaf": "SecureIO: Issue Test Leaf (TEST ONLY)",
                              "SecureIOTestRecoveryKey": "SecureIO: Test Recovery Key (TEST ONLY)"}
