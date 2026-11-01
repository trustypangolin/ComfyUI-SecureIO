"""
Safe output paths: output/<stage>/<mirror_dir>/<filename>.

Copied from ComfyUI-API-DockerCPU common/output_paths.py (commit eb62c00) so this pack
has no cross-pack import. Refuses anything that escapes the output directory or lands
inside the input directory (where the original scans live).
"""

import os
import re

import folder_paths

IF_EXISTS_OPTIONS = ["error", "overwrite", "skip"]

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def clean_relative_dir(value, label):
    value = (value or "").replace("\\", "/").strip()
    if not value:
        return ""
    if value.startswith("/") or _DRIVE_RE.match(value):
        raise ValueError(f"{label} must be a relative folder, got {value!r}")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"{label} must not contain '..': {value!r}")
    return "/".join(parts)


def clean_name_part(value, label):
    value = value or ""
    if "/" in value or "\\" in value or _DRIVE_RE.match(value) or value in (".", ".."):
        raise ValueError(f"{label} must be a plain filename part, got {value!r}")
    return value


def stage_output_path(stage, mirror_dir, filename):
    output_dir = folder_paths.get_output_directory()
    stage = clean_relative_dir(stage, "stage")
    mirror_dir = clean_relative_dir(mirror_dir, "mirror_dir")
    filename = clean_name_part(filename, "filename")
    if not filename:
        raise ValueError("filename is empty")
    target = os.path.abspath(os.path.join(output_dir, stage, mirror_dir, filename))
    if not folder_paths.is_within_directory(output_dir, target):
        raise ValueError(f"Path resolves outside of the output directory: {target}")
    if folder_paths.is_within_directory(folder_paths.get_input_directory(), target):
        raise ValueError(f"Refusing to write under the input directory: {target}")
    return target


def write_atomic(path, data, if_exists):
    """Write bytes via a temp file in the same folder, then move into place. Returns True if written.
    The temp file only ever holds ciphertext."""
    if if_exists not in IF_EXISTS_OPTIONS:
        raise ValueError(f"if_exists must be one of {IF_EXISTS_OPTIONS}, got {if_exists!r}")
    if os.path.exists(path):
        if if_exists == "skip":
            return False
        if if_exists == "error":
            raise FileExistsError(f"Output file already exists: {path} (set if_exists to overwrite or skip)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return True
