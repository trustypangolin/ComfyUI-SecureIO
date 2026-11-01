"""
Encrypted savers (node request 06, SEC2 / SEC3).

Both write output/<stage>/<mirror_dir>/<stem><suffix>.<ext>.ceio, sealed to the client's
X25519 public key (`recipient`). The server can never decrypt what it saved. PNG / MP4 bytes
are built in memory only; prompt, workflow and `restore` metadata live inside the ciphertext.
No preview images are returned to the UI.
"""

import io
import json
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

from . import ceio
from .paths import IF_EXISTS_OPTIONS, clean_name_part, stage_output_path, write_atomic

try:
    from comfy.cli_args import args as _comfy_args
except ImportError:
    _comfy_args = None

CATEGORY = "SecureIO"


def _metadata_allowed(embed_workflow):
    return embed_workflow and not (_comfy_args is not None and _comfy_args.disable_metadata)


def _parse_restore(extra_metadata):
    if not extra_metadata or not extra_metadata.strip():
        return {}
    try:
        value = json.loads(extra_metadata)
    except json.JSONDecodeError as e:
        raise ValueError(f"extra_metadata is not valid JSON: {e}") from e
    return value if isinstance(value, dict) else {"value": value}


def _names(stem, suffix, ext, count):
    stem = clean_name_part(stem, "stem").strip()
    if not stem:
        raise ValueError("stem is empty; wire STEM from the loader or type a name")
    suffix = clean_name_part(suffix, "suffix")
    if count == 1:
        return [f"{stem}{suffix}.{ext}"]
    width = max(2, len(str(count - 1)))
    return [f"{stem}{suffix}_{i:0{width}d}.{ext}" for i in range(count)]


def _check_targets(paths, if_exists):
    if if_exists == "error":
        for path in paths:
            if os.path.exists(path):
                raise FileExistsError(f"Output file already exists: {path} (set if_exists to overwrite or skip)")


def _ui_entry(path):
    sub = os.path.relpath(os.path.dirname(path), folder_paths.get_output_directory()).replace("\\", "/")
    return {"filename": os.path.basename(path), "subfolder": "" if sub == "." else sub, "type": "output"}


COMMON_INPUTS = {
    "recipient": ("STRING", {"default": "", "tooltip": "Client's X25519 public key (base64). Only its private key can decrypt the output."}),
    "stage": ("STRING", {"default": "00_test", "tooltip": "Top-level folder under output, e.g. dataset_prep/01_isolate."}),
    "mirror_dir": ("STRING", {"default": "", "tooltip": "Subfolder under the stage, e.g. MIRROR_DIR from the loader."}),
    "stem": ("STRING", {"default": "", "tooltip": "Filename without extension, e.g. STEM from the loader."}),
    "if_exists": (IF_EXISTS_OPTIONS, {"default": "error"}),
    "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Store prompt + workflow INSIDE the ciphertext."}),
}
OPTIONAL_INPUTS = {
    "suffix": ("STRING", {"default": "", "tooltip": "Appended to the stem, e.g. _p01_head."}),
    "extra_metadata": ("STRING", {"default": "", "multiline": True, "tooltip": "JSON stored (encrypted) as the 'restore' metadata."}),
}


class SaveEncryptedImageWithFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), **COMMON_INPUTS}, "optional": dict(OPTIONAL_INPUTS),
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Saves images as output/<stage>/<mirror_dir>/<stem><suffix>.png.ceio, encrypted to `recipient`. No previews."

    def save(self, images, recipient, stage, mirror_dir, stem, if_exists="error", embed_workflow=True,
             suffix="", extra_metadata="", prompt=None, extra_pnginfo=None):
        pub = ceio.parse_public(recipient)
        restore = {**_parse_restore(extra_metadata), "encrypted": True}
        names = _names(stem, suffix, "png", len(images))
        paths = [stage_output_path(stage, mirror_dir, n + ".ceio") for n in names]
        _check_targets(paths, if_exists)

        info = PngInfo()
        if _metadata_allowed(embed_workflow):
            if prompt is not None:
                info.add_text("prompt", json.dumps(prompt))
            for key, value in (extra_pnginfo or {}).items():
                info.add_text(key, json.dumps(value))
        info.add_text("restore", json.dumps(restore))

        ui = []
        for image, name, path in zip(images, names, paths):
            arr = np.clip(np.round(image.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG", pnginfo=info, compress_level=4)
            meta = {"name": name, "mime": "image/png", "width": int(arr.shape[1]), "height": int(arr.shape[0]),
                    "restore": restore}
            write_atomic(path, ceio.seal(buf.getvalue(), meta, pub), if_exists)
            buf.close()
            ui.append(_ui_entry(path))
        return {"ui": {"ceio": ui}, "result": ("\n".join(paths),)}


class SaveEncryptedVideoWithFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), **COMMON_INPUTS}, "optional": dict(OPTIONAL_INPUTS),
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Saves a video as output/<stage>/<mirror_dir>/<stem><suffix>.mp4.ceio (H.264), encrypted to `recipient`."

    def save(self, video, recipient, stage, mirror_dir, stem, if_exists="error", embed_workflow=True,
             suffix="", extra_metadata="", prompt=None, extra_pnginfo=None):
        from comfy_api.latest import Types

        pub = ceio.parse_public(recipient)
        restore = {**_parse_restore(extra_metadata), "encrypted": True}
        (name,) = _names(stem, suffix, "mp4", 1)
        path = stage_output_path(stage, mirror_dir, name + ".ceio")
        _check_targets([path], if_exists)

        metadata = {}
        if _metadata_allowed(embed_workflow):
            metadata.update(extra_pnginfo or {})
            if prompt is not None:
                metadata["prompt"] = prompt
        metadata["restore"] = restore

        buf = io.BytesIO()   # never a file: the plaintext MP4 exists only in RAM
        video.save_to(buf, format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, metadata=metadata)
        w, h = video.get_dimensions()
        meta = {"name": name, "mime": "video/mp4", "width": int(w), "height": int(h), "restore": restore}
        write_atomic(path, ceio.seal(buf.getvalue(), meta, pub, flags=ceio.FLAG_VIDEO), if_exists)
        buf.close()
        return {"ui": {"ceio": [_ui_entry(path)]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "SaveEncryptedImageWithFilename": SaveEncryptedImageWithFilename,
    "SaveEncryptedVideoWithFilename": SaveEncryptedVideoWithFilename,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveEncryptedImageWithFilename": "Save Encrypted Image With Filename",
    "SaveEncryptedVideoWithFilename": "Save Encrypted Video With Filename",
}
