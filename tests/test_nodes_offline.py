"""Offline test of the savers, using ComfyUI's own modules but a scratch output dir.

Run with the ComfyUI venv python:  python tests/test_nodes_offline.py <ComfyUI dir> <scratch dir>
Checks: only .ceio files written, nothing plaintext anywhere under scratch, decrypts to the same pixels,
metadata inside, video round trip, path safety, if_exists.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from fractions import Fraction

COMFY, SCRATCH = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[:1]
sys.path.insert(0, COMFY)

import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import folder_paths  # noqa: E402

PACK = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
__import__(PACK)   # import as a package so the relative imports work
nodes = sys.modules[PACK + ".nodes"]
ceio = sys.modules[PACK + ".ceio"]


class SaverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(dir=SCRATCH)
        cls.out, cls.inp, cls.tmp = (os.path.join(cls.root, d) for d in ("output", "input", "temp"))
        for d in (cls.out, cls.inp, cls.tmp):
            os.makedirs(d)
        folder_paths.set_output_directory(cls.out)
        folder_paths.set_input_directory(cls.inp)
        folder_paths.set_temp_directory(cls.tmp)
        cls.sk, pub_raw = ceio.generate_keypair()
        import base64
        cls.pub = base64.b64encode(pub_raw).decode()
        g = torch.Generator().manual_seed(0)
        cls.imgs = torch.rand(2, 96, 128, 3, generator=g)

    def all_files(self):
        return [os.path.join(r, f) for r, _, fs in os.walk(self.root) for f in fs]

    def test_image(self):
        res = nodes.SaveEncryptedImageWithFilename().save(
            self.imgs, self.pub, "dataset_prep/01_isolate", "subject01/2000 Jan", "img0001", "error", True,
            suffix="_p03_head", extra_metadata='{"source": "originals/2000 Jan/img0001.jpg"}',
            prompt={"1": {"class_type": "X"}}, extra_pnginfo={"workflow": {"nodes": []}})
        paths = res["result"][0].split("\n")
        self.assertEqual([os.path.basename(p) for p in paths],
                         ["img0001_p03_head_00.png.ceio", "img0001_p03_head_01.png.ceio"])
        self.assertNotIn("images", res["ui"])
        for i, p in enumerate(paths):
            data = Path(p).read_bytes()
            for s in (b"img0001", b"originals", b"IHDR", b"workflow"):
                self.assertNotIn(s, data)
            meta, body = ceio.open_(data, self.sk)
            self.assertEqual(meta["restore"], {"source": "originals/2000 Jan/img0001.jpg", "encrypted": True})
            im = Image.open(io.BytesIO(body))
            want = np.clip(np.round(self.imgs[i].numpy() * 255), 0, 255).astype(np.uint8)
            self.assertTrue(np.array_equal(np.asarray(im), want))
            self.assertIn("prompt", im.text)
            self.assertIn("workflow", im.text)
            self.assertEqual(json.loads(im.text["restore"])["encrypted"], True)
        self.assertTrue(all(f.endswith(".ceio") for f in self.all_files()), self.all_files())
        with self.assertRaises(FileExistsError):
            nodes.SaveEncryptedImageWithFilename().save(self.imgs, self.pub, "dataset_prep/01_isolate",
                                                        "subject01/2000 Jan", "img0001", "error", True,
                                                        suffix="_p03_head")

    def test_no_workflow(self):
        res = nodes.SaveEncryptedImageWithFilename().save(self.imgs[:1], self.pub, "t", "", "nowf", "overwrite",
                                                          False, prompt={"a": 1}, extra_pnginfo={"workflow": {}})
        _, body = ceio.open_(Path(res["result"][0]).read_bytes(), self.sk)
        self.assertNotIn("prompt", Image.open(io.BytesIO(body)).text)

    def test_path_safety(self):
        s = nodes.SaveEncryptedImageWithFilename()
        for stage, mirror, stem in (("../x", "", "a"), ("C:/x", "", "a"), ("t", "../../x", "a"), ("t", "", "a/b")):
            with self.assertRaises(ValueError):
                s.save(self.imgs[:1], self.pub, stage, mirror, stem, "overwrite", True)
        with self.assertRaises(ceio.CeioError):
            s.save(self.imgs[:1], "not-a-key", "t", "", "a", "overwrite", True)

    def test_video(self):
        from comfy_api.latest import InputImpl, Types
        frames = torch.rand(30, 64, 64, 3, generator=torch.Generator().manual_seed(1))
        v = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, frame_rate=Fraction(24)))
        before = set(os.listdir(tempfile.gettempdir()))
        res = nodes.SaveEncryptedVideoWithFilename().save(v, self.pub, "dataset_prep/05_orbit", "subject01",
                                                          "clip", "overwrite", True, suffix="_orbitL",
                                                          prompt={"p": 1})
        new_tmp = [f for f in set(os.listdir(tempfile.gettempdir())) - before if f.endswith((".mp4", ".png"))]
        self.assertEqual(new_tmp, [])
        data = Path(res["result"][0]).read_bytes()
        self.assertEqual(ceio.read_header(data)["flags"], ceio.FLAG_VIDEO)
        meta, body = ceio.open_(data, self.sk)
        self.assertEqual((meta["mime"], meta["width"], meta["height"]), ("video/mp4", 64, 64))
        import av
        with av.open(io.BytesIO(body)) as c:
            self.assertEqual(sum(1 for _ in c.decode(video=0)), 30)
            self.assertIn("restore", c.metadata)


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1], verbosity=2)
