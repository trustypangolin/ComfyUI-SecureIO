# ComfyUI-SecureIO

Encrypted outputs for ComfyUI: the server writes only ciphertext that it can't decrypt itself. Plus an encrypted image loader for inputs, which decrypts with a leaf key kept on the RAM disk, or with a long-lived recovery key.

## Nodes

| Node | Writes |
|---|---|
| `SaveEncryptedImageWithFilename` | `output/<stage>/<mirror_dir>/<stem><suffix>.png.ceio` |
| `SaveEncryptedVideoWithFilename` | `output/<stage>/<mirror_dir>/<stem><suffix>.mp4.ceio` (H.264) |

- **Inputs** are those of [ComfyUI-API-DockerCPU](https://github.com/trustypangolin/ComfyUI-API-DockerCPU)'s `SaveImageWithFilename`, plus `recipient`: the client's X25519 public key, base64.
- **`recovery_recipient`** (optional socket): a second X25519 public key, e.g. the recovery key. When wired, the file is CEIO v2 and either key opens it. Unwired, the file stays v1.
- **What gets encrypted:** the PNG or MP4 is built in memory and sealed. Prompt, workflow and `restore` metadata go inside the ciphertext.
- **What the server keeps:** the job keeps no secret, and no preview images are returned.
- **Decrypting:** `python ceio.py decrypt --key KEYFILE file.png.ceio`, with any recipient's key.

| Loader / tool | Reads |
|---|---|
| `LoadEncryptedImageWithMetadata` | a `.ceio` image (v1 or v2), decrypted in memory |
| `SecureIOInspectCeio` | only the cleartext header: version, flags, each recipient's key_id, and which local key matches it |

- **Key choice:**
  - `(match key_id)`: the key in the key dir (default `R:\leaf`, `SECUREIO_KEY_DIR`) whose key_id is one of the file's recipients. The header holds only those 16-byte fingerprints, no path or name.
  - `(recovery key)`: the recovery key (default `G:\Ai\build\ca\test-root\recovery.ceio.json`, `SECUREIO_RECOVERY_KEY`).
  - a `*.ceio.json` picked in the dropdown, or a wired `key_file` (inside the key dir).
- **`delete_leaf_after_use`** (default on): after a successful decrypt, the leaf key and its `.crt` are overwritten and deleted from the key dir. Savers seal with public keys only, so nothing later needs it. The recovery key and anything outside the key dir are never deleted. Overwriting doesn't guarantee erasure on an SSD; keep leaves on a RAM disk (or in memory only).
- **Cert check before decrypting:** the key's `<name>.crt` must sit next to it, chain to the root cert (`SECUREIO_ROOT_CERT`, default the test root's `root.crt`), be a key-agreement cert and hold the key's own public key. **Cert dates are not enforced**: anyone holding a private key can decrypt outside this node anyway, so ending access means destroying the key.
- **Outputs:** the same 14, in the same order, as DockerCPU's `LoadImageWithMetadata`, so it is a drop-in swap; then `KEY_ID` (the key that opened it) and `INFO`. Names have `.ceio` removed; `SHA256` hashes the decrypted bytes; `ORIGINAL` / `ORIGINAL_SHA256` come only from the provenance sealed in the file.
- **Wired inputs:** `ceio_path` (a `.ceio` in `input/` or `output/`, e.g. a saver's `saved_path`) and `key_file` override the dropdowns; leave the dropdowns on `(use ceio_path)` / `(use key_file)`.
- **What it doesn't cover:** downstream savers and previews still write whatever they normally write. Use the encrypted savers if the outputs must stay sealed too.

| Folder / B2 node | Does |
|---|---|
| `SecureIOEncryptFolder` | **HOST ONLY, test keys.** Seals each image's original bytes in an input folder to a **new leaf of its own** (into the key dir) plus the recovery key → `output/<stage>/<input folder>/<name>.ceio`. |
| `LoadEncryptedImagesFromFolder` | Every `.ceio` under a folder in `output/` or `input/`, decrypted in memory, as lists (one entry per file, like `LoadImagesFromFolderWithMetadata`). Key: `(match key_id)` (each file's own leaf, deleted after use by default) or `(recovery key)`. `REL_DIR` is relative to that folder, for mirrored saving. All files are held in memory at once. |
| `SecureIOUploadToB2` | Uploads a saver's `.ceio` from `output/` to B2 (S3 API; `B2_KEY_ID`, `B2_APP_KEY`, `B2_ENDPOINT` from the environment). |
| `SecureIODownloadFromB2` | Downloads every `.ceio` under a B2 prefix into `output/<stage>/`. |

Both B2 nodes refuse anything that doesn't parse as a CEIO file, in either direction, and refuse an empty prefix.

Workflows in `Photo-Restore/secure/`: `host_only_encrypt_folder_batch_v1` → `vast_b2_save_image_batch_v1` (the worker's job; no previews) → `host_only_decrypt_folder_batch_v1`.

| Test node | Does |
|---|---|
| `SecureIOIssueTestLeaf` | **TEST ONLY.** Issues a new X25519 leaf, signed by the test root (`G:\Ai\build\ca\test-root`, override `SECUREIO_TEST_ROOT`), into the key dir. Its cert dates (7 days) are informational. Outputs `key_file`, `cert_path`, `recipient`, `key_id`, `info`. |
| `SecureIOTestRecoveryKey` | **TEST ONLY.** The recovery key next to the test root, created on first use; after that it reads only `recovery.crt` (public), never the private key or the root key. Outputs `recipient`, `key_id`, `info`. |

Both read the test root key inside ComfyUI, so never point them at a real root. Where the real root and recovery keys live is still to be decided.

Workflows that run on a remote worker must not contain preview nodes (see the project CLAUDE.md).

## Format

`ceio.py` is the CEIO container, usable as a library or a CLI: HPKE X25519 / HKDF-SHA256 / AES-256-GCM sealing a file key, then 64 KiB AES-256-GCM STREAM chunks. v1 has one recipient; v2 has up to 16 (the file key sealed once per recipient), and their key_ids are the only cleartext. Its docstring has the byte layout.

## Tests

`python tests/test_nodes_offline.py <ComfyUI dir> <scratch dir>`, run with the ComfyUI venv. It doesn't need the server.

## Rules

- CPU-only.
- Dependencies: `cryptography>=49`, already in the venv.
- No network, except the two B2 nodes, which only move CEIO ciphertext.
- Never writes plaintext image bytes to disk. (Plaintext output on the host is done by an ordinary saver, e.g. DockerCPU's `SaveImageWithFilename`, never by this pack.)
- Refuses paths outside `output/` or inside `input/`.
