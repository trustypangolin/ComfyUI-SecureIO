"""
Folder-level SecureIO nodes, for the host -> worker -> B2 -> host round trip.

- SecureIOEncryptFolder (HOST ONLY): every image in an input folder is sealed, as its original file bytes, to a
  NEW leaf of its own plus the recovery key (CEIO v2). Leaves go to the key dir (R:\\leaf); the .ceio files to
  output/<stage>/<same subfolders>/<name>.ceio. Reads the TEST root key to sign the leaves.
- LoadEncryptedImagesFromFolder: every .ceio under a folder in input/ or output/, decrypted in memory, as lists
  (one entry per file, like DockerCPU's LoadImagesFromFolderWithMetadata). Each file's key is found by key_id
  ("(match key_id)", the leaf) or is the recovery key; used leaves are deleted by default.
- SecureIOUploadToB2 / SecureIODownloadFromB2: move .ceio files between output/ and a B2 bucket (S3 API,
  credentials B2_KEY_ID / B2_APP_KEY / B2_ENDPOINT from the environment). They only ever move ciphertext:
  anything that doesn't parse as a CEIO file is refused, in both directions.
"""

import datetime as dt
import fnmatch
import hashlib
import io
import mimetypes
import os
import re

import folder_paths
from PIL import Image

from . import ceio
from .loaders import (KEY_MATCH, KEY_RECOVERY, LoadEncryptedImageWithMetadata, _list_keys, key_dir,
                      recovery_key_path)
from .paths import IF_EXISTS_OPTIONS, clean_relative_dir, stage_output_path, write_atomic

CATEGORY = "SecureIO"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_SKIP_DIRS = {".stfolder", ".stversions", "__pycache__"}


def _input_subfolders():
    base = folder_paths.get_input_directory()
    out = ["."]
    for root, dirs, _ in os.walk(base, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("_trash"))
        out += [os.path.relpath(os.path.join(root, d), base).replace("\\", "/") for d in dirs]
    return out


def _files(folder_abs, recursive, keep):
    found = []
    for root, dirs, files in os.walk(folder_abs, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("_trash"))
        found += [os.path.join(root, f) for f in sorted(files) if keep(f)]
        if not recursive:
            break
    return found


def _resolve_folder(root, folder):
    base = folder_paths.get_output_directory() if root == "output" else folder_paths.get_input_directory()
    rel = clean_relative_dir(folder, "folder") if folder not in ("", ".") else ""
    path = os.path.abspath(os.path.join(base, rel))
    if not folder_paths.is_within_directory(base, path) or not os.path.isdir(path):
        raise ValueError(f"folder {folder!r} is not a folder inside {root}/ ({base})")
    return path


# ------------------------------------------------------------------ host only: encrypt a folder

class SecureIOEncryptFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": (_input_subfolders(), {"tooltip": "Input subfolder to encrypt ('.' = input itself). Originals are only read."}),
            "pattern": ("STRING", {"default": "*", "tooltip": "Filename glob, case-insensitive."}),
            "recursive": ("BOOLEAN", {"default": False}),
            "stage": ("STRING", {"default": "secure/encrypted_in", "tooltip": "Output folder; subfolders mirror the input."}),
            "if_exists": (IF_EXISTS_OPTIONS, {"default": "overwrite"}),
            "recovery_recipient": ("STRING", {"forceInput": True, "tooltip": "Recovery public key (from Test Recovery Key)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("stage", "out_folder", "count", "info")
    FUNCTION = "encrypt"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("HOST ONLY, TEST keys. Seals each image's original bytes to a new leaf of its own (into R:\\leaf) "
                   "plus the recovery key. Writes only .ceio files.")

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")   # new leaves every run

    def encrypt(self, folder, pattern, recursive, stage, if_exists, recovery_recipient):
        from .keys import LEAF_CERT_VALIDITY, _load_root, _write_key, _x25519_cert
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

        src = _resolve_folder("input", folder)
        inp = folder_paths.get_input_directory()
        rec_pub = ceio.parse_public(recovery_recipient)
        rec_id = ceio.key_id_for(ceio.public_bytes(rec_pub)).hex()
        pat = (pattern or "*").lower()
        files = _files(src, recursive, lambda f: os.path.splitext(f)[1].lower() in IMAGE_EXTS
                       and fnmatch.fnmatch(f.lower(), pat))
        if not files:
            raise ValueError(f"no images matching {pattern!r} in input/{folder}")
        kdir = key_dir()
        os.makedirs(kdir, exist_ok=True)
        root_key, root_cert = _load_root()
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines = [f"HOST ONLY encrypt: {len(files)} file(s) from input/{folder}, recovery key_id {rec_id}"]
        for path in files:
            with open(path, "rb") as f:
                body = f.read()
            rel_dir = os.path.relpath(os.path.dirname(path), src).replace("\\", "/")
            rel_dir = "" if rel_dir == "." else rel_dir
            name = os.path.basename(path)
            with Image.open(io.BytesIO(body)) as im:
                w, h = im.size
            sk = X25519PrivateKey.generate()
            leaf_id = ceio.key_id_for(ceio.public_bytes(sk.public_key())).hex()
            base = f"{os.path.splitext(name)[0]}_{stamp}_{leaf_id[:8]}"
            cert = _x25519_cert(root_key, root_cert, base, sk, LEAF_CERT_VALIDITY)
            _write_key(os.path.join(kdir, base + ".ceio.json"), os.path.join(kdir, base + ".crt"), sk, cert,
                       "TEST LEAF KEY - RAM disk only")
            meta = {"name": name, "mime": mimetypes.guess_type(name)[0] or "application/octet-stream",
                    "width": w, "height": h,
                    "restore": {"source": os.path.relpath(path, inp).replace("\\", "/"),
                                "source_sha256": hashlib.sha256(body).hexdigest(), "schema": 1,
                                "original": None, "original_sha256": None, "encrypted": True}}
            out = stage_output_path(stage, "/".join(p for p in (folder if folder != "." else "", rel_dir) if p),
                                    name + ".ceio")
            write_atomic(out, ceio.seal(body, meta, sk.public_key(), extra_recipients=(rec_pub,)), if_exists)
            del sk, body
            lines.append(f"{os.path.relpath(out, folder_paths.get_output_directory())}  <- leaf {leaf_id} "
                         f"({base}.ceio.json)")
        out_stage = clean_relative_dir(stage, "stage")
        info = "\n".join(lines)
        return {"ui": {"text": [info]},
                "result": (out_stage, os.path.join(folder_paths.get_output_directory(), out_stage), len(files), info)}


# ------------------------------------------------------------------ worker or host: decrypt a folder (lists)

class LoadEncryptedImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "root": (["output", "input"], {"default": "output"}),
            "folder": ("STRING", {"default": "secure/encrypted_in", "tooltip": "Folder under root holding .ceio files."}),
            "recursive": ("BOOLEAN", {"default": True}),
            "key": ([KEY_MATCH, KEY_RECOVERY], {"tooltip": "(match key_id): each file's own leaf in R:\\leaf. (recovery key): the recovery key."}),
            "delete_leaf_after_use": ("BOOLEAN", {"default": True, "tooltip": "Delete each leaf after its file is decrypted (never the recovery key)."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "STRING", "STRING",
                    "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "BASE_FILENAME", "STEM", "REL_DIR", "SHA256", "width", "height", "ORIGINAL",
                    "ORIGINAL_SHA256", "KEY_ID", "INFO")
    OUTPUT_IS_LIST = (True,) * 12
    FUNCTION = "execute"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Decrypts every .ceio under a folder, in memory, as lists (one entry per file). REL_DIR is the "
                   "file's subfolder relative to that folder, for mirrored saving.")

    @classmethod
    def IS_CHANGED(cls, root, folder, recursive, key, delete_leaf_after_use):
        try:
            files = _files(_resolve_folder(root, folder), recursive, lambda f: f.endswith(".ceio"))
        except ValueError as e:
            return str(e)
        m = hashlib.sha256(f"{key}|{delete_leaf_after_use}|{'|'.join(_list_keys())}".encode())
        for p in files:
            st = os.stat(p)
            m.update(f"{p}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        return m.hexdigest()

    def execute(self, root, folder, recursive, key, delete_leaf_after_use):
        src = _resolve_folder(root, folder)
        files = _files(src, recursive, lambda f: f.endswith(".ceio"))
        if not files:
            raise ValueError(f"no .ceio files in {root}/{folder}")
        single = LoadEncryptedImageWithMetadata()
        out = tuple([] for _ in self.RETURN_TYPES)
        for path in files:
            r = single.execute("", key, delete_leaf_after_use, ceio_path=path)
            rel_dir = os.path.relpath(os.path.dirname(path), src).replace("\\", "/")
            values = (r[0], r[1], r[4], r[10], "" if rel_dir == "." else rel_dir, r[11], r[7], r[6], r[12], r[13],
                      r[14], r[15])
            for lst, v in zip(out, values):
                lst.append(v)
        return out


# ------------------------------------------------------------------ B2 (ciphertext only)

def _b2():
    import boto3
    endpoint = os.environ.get("B2_ENDPOINT", "")
    m = re.search(r"s3\.([^.]+)\.backblazeb2", endpoint)
    if not (endpoint and os.environ.get("B2_KEY_ID") and os.environ.get("B2_APP_KEY") and m):
        raise ValueError("B2_ENDPOINT / B2_KEY_ID / B2_APP_KEY are not set in ComfyUI's environment")
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=os.environ["B2_KEY_ID"],
                        aws_secret_access_key=os.environ["B2_APP_KEY"], region_name=m.group(1))


def _bucket(bucket):
    bucket = (bucket or "").strip() or os.environ.get("SECUREIO_B2_BUCKET", "").strip()
    if not bucket:
        raise ValueError("no bucket: set the bucket widget or SECUREIO_B2_BUCKET")
    return bucket


def _clean_prefix(prefix):
    prefix = clean_relative_dir(prefix, "prefix")
    if not prefix:
        raise ValueError("prefix must not be empty (refusing to work on the bucket root)")
    return prefix + "/"


def _require_ceio(data, label):
    try:
        ceio.read_header(data)
    except ceio.CeioError as e:
        raise ValueError(f"{label} is not a CEIO file ({e}); refusing to move anything that isn't ciphertext") from None


class SecureIOUploadToB2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "saved_path": ("STRING", {"forceInput": True, "tooltip": "A saver's saved_path (one .ceio in output/)."}),
            "bucket": ("STRING", {"default": "", "tooltip": "B2 bucket. Empty = SECUREIO_B2_BUCKET from the environment."}),
            "prefix": ("STRING", {"default": "secureio-test/vast_out", "tooltip": "Key prefix; the file keeps its path relative to strip_stage."}),
            "strip_stage": ("STRING", {"default": "secure/vast_out", "tooltip": "Output folder removed from the local path to make the key."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("b2_uri",)
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Uploads one .ceio from output/ to B2 (S3 API). Refuses anything that isn't CEIO ciphertext."

    def upload(self, saved_path, bucket, prefix, strip_stage):
        bucket = _bucket(bucket)
        uris = []
        for p in [s for s in saved_path.split("\n") if s.strip()]:
            path = os.path.abspath(p.strip())
            out_dir = folder_paths.get_output_directory()
            if not folder_paths.is_within_directory(out_dir, path) or not path.endswith(".ceio"):
                raise ValueError(f"only .ceio files inside output/ can be uploaded: {p!r}")
            with open(path, "rb") as f:
                data = f.read()
            _require_ceio(data, os.path.basename(path))
            base = os.path.join(out_dir, clean_relative_dir(strip_stage, "strip_stage"))
            rel = os.path.relpath(path, base if folder_paths.is_within_directory(base, path) else out_dir)
            key = _clean_prefix(prefix) + rel.replace("\\", "/")
            _b2().put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/octet-stream")
            uris.append(f"b2://{bucket}/{key}")
        text = "\n".join(uris)
        return {"ui": {"text": [text]}, "result": (text,)}


class SecureIODownloadFromB2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "bucket": ("STRING", {"default": "", "tooltip": "B2 bucket. Empty = SECUREIO_B2_BUCKET from the environment."}),
            "prefix": ("STRING", {"default": "secureio-test/vast_out"}),
            "stage": ("STRING", {"default": "secure/from_b2", "tooltip": "Output folder to download into; keys keep their sub-path."}),
            "if_exists": (IF_EXISTS_OPTIONS, {"default": "overwrite"}),
        }}

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("stage", "count", "info")
    FUNCTION = "download"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Downloads every .ceio under a B2 prefix into output/<stage>/. Refuses anything that isn't CEIO."

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")   # the bucket can change at any time

    def download(self, bucket, prefix, stage, if_exists):
        bucket = _bucket(bucket)
        pre = _clean_prefix(prefix)
        s3 = _b2()
        keys = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=pre):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".ceio")]
        if not keys:
            raise ValueError(f"no .ceio objects under b2://{bucket}/{pre}")
        lines = [f"{len(keys)} .ceio object(s) from b2://{bucket}/{pre}"]
        for k in sorted(keys):
            rel = k[len(pre):]
            sub, name = (rel.rsplit("/", 1) if "/" in rel else ("", rel))
            data = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
            _require_ceio(data, k)
            out = stage_output_path(stage, clean_relative_dir(sub, "key path"), name)
            write_atomic(out, data, if_exists)
            lines.append(f"{k} -> {os.path.relpath(out, folder_paths.get_output_directory())}")
        info = "\n".join(lines)
        return {"ui": {"text": [info]}, "result": (clean_relative_dir(stage, "stage"), len(keys), info)}


NODE_CLASS_MAPPINGS = {
    "SecureIOEncryptFolder": SecureIOEncryptFolder,
    "LoadEncryptedImagesFromFolder": LoadEncryptedImagesFromFolder,
    "SecureIOUploadToB2": SecureIOUploadToB2,
    "SecureIODownloadFromB2": SecureIODownloadFromB2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SecureIOEncryptFolder": "SecureIO: Encrypt Folder (HOST ONLY, leaf per image + recovery)",
    "LoadEncryptedImagesFromFolder": "Load Encrypted Images From Folder",
    "SecureIOUploadToB2": "SecureIO: Upload .ceio to B2",
    "SecureIODownloadFromB2": "SecureIO: Download .ceio from B2",
}
