"""
Encrypted image loader: the input-side counterpart of SaveEncryptedImageWithFilename.

LoadEncryptedImageWithMetadata reads a .ceio image (v1 or v2), decrypts it in memory and returns the same
14 outputs, in the same order, as DockerCPU's LoadImageWithMetadata (a drop-in swap), plus KEY_ID and INFO.

- The plaintext image only ever exists in RAM; nothing decrypted is written by this node.
- Which key:
  - "(match key_id)": the key in the key dir (default R:\\leaf, SECUREIO_KEY_DIR) whose key_id is one of the
    file header's recipients. The header holds only those 16-byte fingerprints, no names or paths.
  - "(recovery key)": the long-lived recovery key (default <test root>\\recovery.ceio.json,
    SECUREIO_RECOVERY_KEY). Opens any file sealed with that key as a recipient, after the leaf is gone.
  - a *.ceio.json picked in the dropdown, or a wired key_file (must be inside the key dir).
- Cert check before decrypting: the key's `<name>.crt` must sit next to it, chain to the root cert
  (SECUREIO_ROOT_CERT, default the test root), be a key-agreement cert and hold the key's own public key.
  Cert dates are NOT enforced: anyone holding a private key can decrypt anyway, so expiry is a matter of
  destroying keys, not of this node.
- delete_leaf_after_use (default on): after a successful decrypt the leaf key file and its .crt are
  overwritten and deleted from the key dir. Savers seal with public keys only, so nothing later in the
  workflow needs the leaf private key. The recovery key and anything outside the key dir are never deleted.
- BASE_FILENAME / STEM / REL_DIR describe the plaintext name (".ceio" removed). SHA256 hashes the decrypted
  bytes. ORIGINAL / ORIGINAL_SHA256 come only from the provenance sealed inside the file.

SecureIOInspectCeio shows what a .ceio reveals without any key (version, recipients' key_ids).
"""

import base64
import glob
import hashlib
import io
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngImageFile

import folder_paths

from . import ceio

CATEGORY = "SecureIO"
# Fixed first choices in the dropdowns. "(use ...)" = take the wired input; keeps the dropdowns valid when
# the folder is empty or the file/key is made in the same run.
IMAGE_WIRED = "(use ceio_path)"
KEY_MATCH = "(match key_id)"
KEY_RECOVERY = "(recovery key)"
KEY_WIRED = "(use key_file)"
NO_KEYS = KEY_WIRED   # older name
_PSEUDO_KEYS = (KEY_MATCH, KEY_RECOVERY, KEY_WIRED)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_EXCLUDED_DIRS = {".stfolder", ".stversions", "_trash", "__pycache__"}


def key_dir():
    return os.path.abspath(os.environ.get("SECUREIO_KEY_DIR", r"R:\leaf"))


def _test_root():
    return os.environ.get("SECUREIO_TEST_ROOT", r"G:\Ai\build\ca\test-root")


def root_cert_path():
    return os.path.abspath(os.environ.get("SECUREIO_ROOT_CERT", os.path.join(_test_root(), "root.crt")))


def recovery_key_path():
    return os.path.abspath(os.environ.get("SECUREIO_RECOVERY_KEY", os.path.join(_test_root(), "recovery.ceio.json")))


def _list_keys():
    d = key_dir()
    found = sorted(f for f in os.listdir(d) if f.endswith(".ceio.json") and os.path.isfile(os.path.join(d, f))) \
        if os.path.isdir(d) else []
    return [KEY_MATCH, KEY_RECOVERY, KEY_WIRED] + found


def _key_file_id(path):
    with open(path, encoding="utf-8") as f:
        pub = json.load(f).get("public")
    raw = base64.b64decode(pub) if pub else ceio.public_bytes(ceio._load_key(path).public_key())
    return ceio.key_id_for(raw).hex()


def _match_key(key_ids):
    """The first key file in the key dir whose public key is one of the file's recipients."""
    for path in sorted(glob.glob(os.path.join(key_dir(), "*.ceio.json"))):
        try:
            if _key_file_id(path) in key_ids:
                return path
        except (OSError, ValueError, KeyError):
            continue
    raise ceio.CeioError(f"no key in {key_dir()} matches the file's recipients {key_ids} (sealed to a key not "
                         "on this machine, or the leaf has been deleted; try '(recovery key)')")


def check_cert(key_path):
    """Refuse unless <key>.crt chains to the root and is a key-agreement cert for this key (dates not checked).
    Returns a one-line summary for INFO."""
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    crt = key_path[:-len(".ceio.json")] + ".crt"
    name = os.path.basename(key_path)
    if not os.path.isfile(crt):
        raise ceio.CeioError(f"no cert next to {name} ({os.path.basename(crt)}); refusing an unverifiable key")
    with open(crt, "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    with open(root_cert_path(), "rb") as f:
        root = x509.load_pem_x509_certificate(f.read())
    try:
        cert.verify_directly_issued_by(root)
    except Exception as e:   # InvalidSignature, ValueError, TypeError
        raise ceio.CeioError(f"{os.path.basename(crt)} does not chain to the root {root_cert_path()}: {e}") from None
    if cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca or \
            not cert.extensions.get_extension_for_class(x509.KeyUsage).value.key_agreement:
        raise ceio.CeioError(f"{os.path.basename(crt)} is not a key-agreement cert")
    pk = cert.public_key()
    if not isinstance(pk, X25519PublicKey) or ceio.key_id_for(ceio.public_bytes(pk)).hex() != _key_file_id(key_path):
        raise ceio.CeioError(f"{os.path.basename(crt)} is for a different key than {name}")
    return f"{cert.subject.rfc4514_string()} chains to root"


def wipe_leaf(key_path):
    """Delete a used leaf key (+ its .crt): overwrite with zeros, then remove. Only files directly in the key
    dir, never the recovery key. Returns a line for INFO. Note: on an SSD the overwrite doesn't guarantee the
    old bytes are gone; that's why leaves belong on a RAM disk (or in memory only)."""
    key_path = os.path.abspath(key_path)
    if os.path.dirname(key_path) != key_dir() or key_path == recovery_key_path():
        return f"kept ({os.path.basename(key_path)} is not a leaf in {key_dir()})"
    done = []
    for p in (key_path, key_path[:-len(".ceio.json")] + ".crt"):
        if os.path.isfile(p):
            size = os.path.getsize(p)
            with open(p, "r+b") as f:
                f.write(b"\0" * size)
                f.flush()
                os.fsync(f.fileno())
            os.remove(p)
            done.append(os.path.basename(p))
    return f"leaf deleted from {key_dir()}: {', '.join(done) or 'nothing found'}"


def _list_ceio_inputs():
    base = folder_paths.get_input_directory()
    found = []
    for root, dirs, files in os.walk(base, followlinks=True):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS and not d.startswith("_trash")]
        for f in files:
            if f.endswith(".ceio"):
                found.append(os.path.relpath(os.path.join(root, f), base).replace("\\", "/"))
    return [IMAGE_WIRED] + sorted(found)


def _resolve_input(image):
    if not image or image == IMAGE_WIRED:
        raise ValueError("no image selected; pick a .ceio file or wire ceio_path")
    base = folder_paths.get_input_directory()
    path = os.path.abspath(os.path.join(base, image))
    if not folder_paths.is_within_directory(base, path):
        raise ValueError(f"image must be inside the input directory: {image!r}")
    if not path.endswith(".ceio"):
        raise ValueError(f"image must be a .ceio file: {image!r}")
    return path


def _resolve_key(key):
    if not key or key in _PSEUDO_KEYS:
        raise ValueError(f"no key selected; pick a leaf key (*.ceio.json) from {key_dir()} or wire key_file")
    d = key_dir()
    path = os.path.abspath(os.path.join(d, key))
    if os.path.dirname(path) != d or not path.endswith(".ceio.json"):
        raise ValueError(f"key must be a *.ceio.json file directly inside {d}: {key!r}")
    return path


def _png_metadata(img):
    prompt, metadata = {}, {}
    if isinstance(img, PngImageFile):
        for k, v in img.info.items():
            if k in ("prompt", "workflow", "restore"):
                try:
                    metadata[k] = json.loads(v)
                except (TypeError, ValueError):
                    metadata[k] = v
            elif isinstance(v, str):
                metadata[k] = v
        if isinstance(metadata.get("prompt"), dict):
            prompt = metadata["prompt"]
    return prompt, metadata


def _to_tensors(img):
    if img.mode == "RGBA":
        rgb = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img).convert("RGB")
    else:
        rgb = img.convert("RGB")
    image = torch.from_numpy(np.asarray(rgb).astype(np.float32) / 255.0)[None,]
    if "A" in img.getbands():
        mask = 1.0 - torch.from_numpy(np.array(img.getchannel("A")).astype(np.float32) / 255.0)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return image, mask.unsqueeze(0)


def _carried_original(restore):
    original, sha = restore.get("original"), restore.get("original_sha256")
    if isinstance(original, str) and isinstance(sha, str) and original and _SHA_RE.match(sha.lower()):
        original = original.replace("\\", "/").strip("/")
        if original and ".." not in original.split("/") and not re.match(r"^[A-Za-z]:", original):
            return original, sha.lower()
    return "", ""


def _resolve_ceio_path(value):
    """A wired path (e.g. a saver's saved_path): one .ceio file inside the input or output folder."""
    value = (value or "").strip()
    if "\n" in value:
        raise ValueError("ceio_path holds several paths; wire a single file")
    path = os.path.abspath(value)
    if not path.endswith(".ceio"):
        raise ValueError(f"ceio_path must be a .ceio file: {value!r}")
    for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
        if folder_paths.is_within_directory(base, path):
            return path, base
    raise ValueError(f"ceio_path must be inside the input or output folder: {value!r}")


class LoadEncryptedImageWithMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_list_ceio_inputs(), {"tooltip": "A .ceio file under the input folder (ignored when ceio_path is wired)."}),
                "key": (_list_keys(), {"tooltip": "(match key_id): the R:\\leaf key that is one of the file's recipients. (recovery key): the long-lived recovery key. Or pick a *.ceio.json. Ignored when key_file is wired. The key's .crt must chain to the root."}),
                "delete_leaf_after_use": ("BOOLEAN", {"default": True, "tooltip": "After a successful decrypt, delete the leaf key file and its .crt from the key dir (it isn't needed again: savers seal with public keys). Never touches the recovery key or anything outside the key dir."}),
            },
            "optional": {
                "ceio_path": ("STRING", {"forceInput": True, "tooltip": "Overrides image: a .ceio path in input/ or output/, e.g. a saver's saved_path."}),
                "key_file": ("STRING", {"forceInput": True, "tooltip": "Overrides key: a *.ceio.json name or path inside the key dir, e.g. from Issue Test Leaf."}),
            },
        }

    # The first 14 match LoadImageWithMetadata, so links carry over; new outputs are only ever appended.
    RETURN_TYPES = ("IMAGE", "MASK", "JSON", "METADATA_RAW", "STRING", "STRING", "INT", "INT", "FLOAT",
                    "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "prompt", "Metadata RAW", "BASE_FILENAME", "BASE_DIR", "height", "width",
                    "megapixels", "REL_DIR", "STEM", "SHA256", "ORIGINAL", "ORIGINAL_SHA256", "KEY_ID", "INFO")
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decrypts a .ceio image in memory with a leaf key from R:\\leaf or the recovery key. The first 14 "
                   "outputs match LoadImageWithMetadata; KEY_ID is the key_id that opened it, INFO a summary.")

    def execute(self, image="", key="", delete_leaf_after_use=True, ceio_path=None, key_file=None):
        if ceio_path:
            path, base = _resolve_ceio_path(ceio_path)
        else:
            path, base = _resolve_input(image), folder_paths.get_input_directory()
        with open(path, "rb") as f:
            data = f.read()
        header = ceio.read_header(data)
        if header["flags"] & ceio.FLAG_VIDEO:
            raise ValueError(f"{os.path.basename(path)} holds a video, not an image")
        if key_file:
            key_path, how = _resolve_key(key_file), "wired key_file"
        elif key == KEY_MATCH:
            key_path, how = _match_key(header["key_ids"]), "matched a recipient key_id in the key dir"
        elif key == KEY_RECOVERY:
            key_path, how = recovery_key_path(), "recovery key"
            if not os.path.isfile(key_path):
                raise ceio.CeioError(f"no recovery key at {key_path}")
        else:
            key_path, how = _resolve_key(key), "picked in dropdown"
        cert_line = check_cert(key_path)          # chain / usage / same key: refuse before decrypting
        private_key = ceio._load_key(key_path)
        key_pub_id = ceio.key_id_for(ceio.public_bytes(private_key.public_key())).hex()
        if key_pub_id not in header["key_ids"]:
            raise ceio.CeioError(f"{os.path.basename(path)} was sealed to key_id(s) {header['key_ids']}, "
                                 f"but {os.path.basename(key_path)} is key_id {key_pub_id}")
        meta, body = ceio.open_(data, private_key)

        img = Image.open(io.BytesIO(body))
        prompt, metadata = _png_metadata(img)
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        tensor, mask = _to_tensors(img)

        plain_name = os.path.basename(path)[:-len(".ceio")]
        base_dir = os.path.dirname(path)
        rel = os.path.relpath(base_dir, base).replace("\\", "/")
        rel_dir = "" if rel == "." else rel
        restore = meta.get("restore") if isinstance(meta.get("restore"), dict) else {}
        metadata["fileinfo"] = {"filename": path.replace("\\", "/"), "base_filename": plain_name,
                                "resolution": f"{width}x{height}", "encrypted": True,
                                "key_id": key_pub_id, "recipients": header["key_ids"]}
        sha = hashlib.sha256(body).hexdigest()
        del private_key
        wiped = wipe_leaf(key_path) if delete_leaf_after_use else "kept (delete_leaf_after_use off)"
        info = "\n".join([
            "decrypted in memory (nothing plaintext written)",
            f"file      : {path}",
            f"header    : CEIO v{header['version']}, {header['recipients']} recipient(s), flags {header['flags']}",
            *[f"recipient : {k}{'   <== opened with this key' if k == key_pub_id else ''}" for k in header["key_ids"]],
            f"key file  : {key_path} ({how})",
            f"cert      : {cert_line}",
            f"after use : {wiped}",
            f"image     : {meta.get('mime')} {width}x{height}, sha256 {sha}",
            f"restore   : {json.dumps(restore, ensure_ascii=False)}",
        ])
        return (tensor, mask, prompt, metadata, plain_name, base_dir, height, width,
                round(width * height / 1_000_000.0, 4), rel_dir, os.path.splitext(plain_name)[0],
                sha, *_carried_original(restore), key_pub_id, info)

    @classmethod
    def IS_CHANGED(cls, image="", key="", **_linked):
        # Wired ceio_path/key_file aren't known here; ComfyUI re-runs the node when they change anyway.
        # Includes the key dir listing so deleting/adding a leaf re-runs "(match key_id)".
        m = hashlib.sha256((key or "").encode())
        m.update("|".join(_list_keys()).encode())
        try:
            with open(_resolve_input(image), "rb") as f:
                m.update(f.read())
        except (OSError, ValueError):
            pass
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, key):
        # The "(...)" choices (or empty) defer to wired inputs / lookups at run time, so they're checked in
        # execute() instead of here.
        try:
            if image and image != IMAGE_WIRED:
                path = _resolve_input(image)
                if not os.path.isfile(path):
                    return f"Invalid image file: {image}"
            if key and key not in _PSEUDO_KEYS:
                _resolve_key(key)
        except ValueError as e:
            return str(e)
        return True


class SecureIOInspectCeio:
    """What a .ceio reveals without any key: format version, flags and each recipient's 16-byte key_id."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ceio_path": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("info", "key_ids")
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Reads only the cleartext header of a .ceio (no key): version, flags, recipients' key_ids."

    def inspect(self, ceio_path):
        path, _ = _resolve_ceio_path(ceio_path)
        with open(path, "rb") as f:
            head = f.read(ceio.HEADER_V2.size + (ceio.KEY_ID_LEN + ceio.HPKE_BLOB_LEN) * ceio.MAX_RECIPIENTS + ceio.TAG)
        h = ceio.read_header(head)
        local = {}
        for kp in glob.glob(os.path.join(key_dir(), "*.ceio.json")) + [recovery_key_path()]:
            try:
                local[_key_file_id(kp)] = kp
            except (OSError, ValueError, KeyError):
                pass
        lines = [f"header only (no key used): {path}",
                 f"CEIO v{h['version']}, {h['recipients']} recipient(s), flags {h['flags']}, "
                 f"{os.path.getsize(path)} bytes"]
        for k in h["key_ids"]:
            lines.append(f"recipient : {k}   -> {local.get(k, 'no matching key on this host')}")
        return ("\n".join(lines), "\n".join(h["key_ids"]))


NODE_CLASS_MAPPINGS = {"LoadEncryptedImageWithMetadata": LoadEncryptedImageWithMetadata,
                       "SecureIOInspectCeio": SecureIOInspectCeio}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadEncryptedImageWithMetadata": "Load Encrypted Image With Metadata",
                              "SecureIOInspectCeio": "SecureIO: Inspect .ceio Header"}
