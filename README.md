# ComfyUI-SecureIO

Encrypted outputs for ComfyUI: the server writes only ciphertext that it can't decrypt itself.

## Nodes

| Node | Writes |
|---|---|
| `SaveEncryptedImageWithFilename` | `output/<stage>/<mirror_dir>/<stem><suffix>.png.ceio` |
| `SaveEncryptedVideoWithFilename` | `output/<stage>/<mirror_dir>/<stem><suffix>.mp4.ceio` (H.264) |

- **Inputs** are those of [ComfyUI-API-DockerCPU](https://github.com/trustypangolin/ComfyUI-API-DockerCPU)'s `SaveImageWithFilename`, plus `recipient`: the client's X25519 public key, base64.
- **What gets encrypted:** the PNG or MP4 is built in memory and sealed to `recipient`. Prompt, workflow and `restore` metadata go inside the ciphertext.
- **What the server keeps:** the job keeps no secret, and no preview images are returned.
- **Decrypting:** `python ceio.py decrypt --key KEYFILE file.png.ceio`.

## Format

`ceio.py` is the CEIO v1 container, usable as a library or a CLI: HPKE X25519 / HKDF-SHA256 / AES-256-GCM sealing a file key, then 64 KiB AES-256-GCM STREAM chunks. Its docstring has the byte layout.

## Tests

`python tests/test_nodes_offline.py <ComfyUI dir> <scratch dir>`, run with the ComfyUI venv. It doesn't need the server.

## Rules

- CPU-only.
- Dependencies: `cryptography>=49`, already in the venv.
- No network.
- Never writes plaintext image bytes to disk.
- Refuses paths outside `output/` or inside `input/`.
