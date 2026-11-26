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
loaders = sys.modules[PACK + ".loaders"]
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


def run_loader(*args, **kwargs):
    """LoadEncryptedImageWithMetadata.execute that keeps the leaf (tests reuse keys); wipe tests pass True."""
    kwargs.setdefault("delete_leaf_after_use", False)
    return loaders.LoadEncryptedImageWithMetadata().execute(*args, **kwargs)


def make_ca(ca_dir):
    """Scratch P-256 root in ca_dir (root.key + root.crt); returns (key, cert)."""
    import datetime as dt
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    os.makedirs(ca_dir, exist_ok=True)
    k = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "scratch root")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=1))
            .not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(k, hashes.SHA256()))
    Path(ca_dir, "root.key").write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption()))
    Path(ca_dir, "root.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return k, cert


def make_leaf(ca, keys_dir, name, after_seconds=86400, cert=True):
    """X25519 leaf <name>.ceio.json (+ <name>.crt signed by ca, expiring after_seconds from now; may be < 0)."""
    import base64
    import datetime as dt
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization as s
    from cryptography.x509.oid import NameOID
    root_key, root_cert = ca
    sk, pub_raw = ceio.generate_keypair()
    raw = sk.private_bytes(s.Encoding.Raw, s.PrivateFormat.Raw, s.NoEncryption())
    pub = base64.b64encode(pub_raw).decode()
    Path(keys_dir, f"{name}.ceio.json").write_text(json.dumps({"private": base64.b64encode(raw).decode(),
                                                              "public": pub}))
    if cert:
        now = dt.datetime.now(dt.timezone.utc)
        leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
                .issuer_name(root_cert.subject).public_key(sk.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - dt.timedelta(minutes=10)).not_valid_after(now + dt.timedelta(seconds=after_seconds))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                             data_encipherment=False, key_agreement=True, key_cert_sign=False,
                                             crl_sign=False, encipher_only=False, decipher_only=False), critical=True)
                .sign(root_key, hashes.SHA256()))
        Path(keys_dir, f"{name}.crt").write_bytes(leaf.public_bytes(s.Encoding.PEM))
    return pub


class LoaderTest(unittest.TestCase):
    """Saver -> move into input/ -> LoadEncryptedImageWithMetadata, with a scratch key dir and scratch root."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(dir=SCRATCH)
        cls.out, cls.inp, cls.keys, cls.ca_dir = (os.path.join(cls.root, d) for d in ("output", "input", "keys", "ca"))
        for d in (cls.out, cls.inp, cls.keys):
            os.makedirs(d)
        folder_paths.set_output_directory(cls.out)
        folder_paths.set_input_directory(cls.inp)
        os.environ["SECUREIO_KEY_DIR"] = cls.keys
        os.environ["SECUREIO_TEST_ROOT"] = cls.ca_dir
        os.environ.pop("SECUREIO_ROOT_CERT", None)
        cls.ca = make_ca(cls.ca_dir)
        cls.pub = make_leaf(cls.ca, cls.keys, "good")
        make_leaf(cls.ca, cls.keys, "other")
        cls.img = torch.rand(1, 40, 56, 3, generator=torch.Generator().manual_seed(2))
        res = nodes.SaveEncryptedImageWithFilename().save(
            cls.img, cls.pub, "x", "", "scan01", "overwrite", True,
            original="1990 Album/scan01.jpg", original_sha256="ab" * 32, require_original=True,
            prompt={"1": {"class_type": "X"}})
        os.makedirs(os.path.join(cls.inp, "people", "ben"))
        cls.rel = "people/ben/scan01.png.ceio"
        os.replace(res["result"][0], os.path.join(cls.inp, cls.rel))

    def seal(self, pub, stem):
        return nodes.SaveEncryptedImageWithFilename().save(self.img, pub, "00_test", "sec_e2e", stem,
                                                           "overwrite", True)["result"][0]

    def test_round_trip(self):
        L = loaders.LoadEncryptedImageWithMetadata
        self.assertIn(self.rel, L.INPUT_TYPES()["required"]["image"][0])
        self.assertEqual(L.INPUT_TYPES()["required"]["key"][0][:3],
                         [loaders.KEY_MATCH, loaders.KEY_RECOVERY, loaders.KEY_WIRED])
        self.assertIn("good.ceio.json", L.INPUT_TYPES()["required"]["key"][0])
        self.assertEqual(L.INPUT_TYPES()["required"]["image"][0][0], loaders.IMAGE_WIRED)
        self.assertIs(L.VALIDATE_INPUTS(self.rel, "good.ceio.json"), True)
        out = run_loader(self.rel, "good.ceio.json")
        self.assertEqual(len(out), len(L.RETURN_TYPES))
        image, mask, prompt, meta, base, _bdir, h, w, mp, rel_dir, stem, sha, orig, orig_sha, key_id, info = out
        self.assertEqual(key_id, ceio.key_id_for(__import__("base64").b64decode(self.pub)).hex())
        self.assertIn(key_id, info)
        self.assertIn("chains to root", info)
        want = np.clip(np.round(self.img[0].numpy() * 255), 0, 255).astype(np.uint8)
        self.assertTrue(np.array_equal(np.round(image[0].numpy() * 255).astype(np.uint8), want))
        self.assertEqual((h, w, base, rel_dir, stem), (40, 56, "scan01.png", "people/ben", "scan01"))
        self.assertEqual(prompt, {"1": {"class_type": "X"}})
        self.assertEqual((orig, orig_sha), ("1990 Album/scan01.jpg", "ab" * 32))
        self.assertEqual(len(sha), 64)
        self.assertTrue(meta["fileinfo"]["encrypted"])
        files = [os.path.join(r, f) for r, _, fs in os.walk(self.root) for f in fs]
        self.assertFalse([f for f in files if f.endswith((".png", ".jpg"))], files)

    def test_match_key_id(self):
        out = run_loader(self.rel, loaders.KEY_MATCH)
        self.assertIn("good.ceio.json (matched a recipient key_id in the key dir)", out[15])
        lone = make_leaf(self.ca, tempfile.mkdtemp(dir=SCRATCH), "elsewhere")   # key not in the key dir
        with self.assertRaisesRegex(ceio.CeioError, "no key in"):
            run_loader("", loaders.KEY_MATCH,
                                                             ceio_path=self.seal(lone, "elsewhere"))

    def test_recovery(self):
        """Seal to a leaf + recovery key, delete the leaf, then only the recovery key opens it."""
        rec_dir = tempfile.mkdtemp(dir=SCRATCH)
        os.environ["SECUREIO_RECOVERY_KEY"] = os.path.join(rec_dir, "recovery.ceio.json")
        try:
            rec_pub = make_leaf(self.ca, rec_dir, "recovery", after_seconds=3650 * 86400)
            leaf_pub = make_leaf(self.ca, self.keys, "temp1")
            path = nodes.SaveEncryptedImageWithFilename().save(
                self.img, leaf_pub, "00_test", "sec_rec", "rec", "overwrite", True,
                recovery_recipient=rec_pub)["result"][0]
            data = Path(path).read_bytes()
            h = ceio.read_header(data)
            ids = [ceio.key_id_for(__import__("base64").b64decode(k)).hex() for k in (leaf_pub, rec_pub)]
            self.assertEqual((h["version"], h["key_ids"]), (2, ids))
            L = loaders.LoadEncryptedImageWithMetadata
            self.assertIn("temp1.ceio.json", run_loader("", loaders.KEY_MATCH, ceio_path=path)[15])
            for f in ("temp1.ceio.json", "temp1.crt"):
                os.remove(os.path.join(self.keys, f))
            with self.assertRaisesRegex(ceio.CeioError, "no key in"):
                run_loader("", loaders.KEY_MATCH, ceio_path=path)
            out = run_loader("", loaders.KEY_RECOVERY, ceio_path=path)
            self.assertEqual(out[14], ids[1])
            self.assertIn("recovery key", out[15])
            want = np.clip(np.round(self.img[0].numpy() * 255), 0, 255).astype(np.uint8)
            self.assertTrue(np.array_equal(np.round(out[0][0].numpy() * 255).astype(np.uint8), want))
            with self.assertRaisesRegex(ceio.CeioError, "not one of the file's recipients"):
                ceio.open_(data, ceio._load_key(os.path.join(self.keys, "good.ceio.json")))
            bad = bytearray(data)
            bad[12] ^= 1                                   # flip a bit in the first recipient's key_id
            with self.assertRaises(ceio.CeioError):
                ceio.open_(bytes(bad), ceio._load_key(os.environ["SECUREIO_RECOVERY_KEY"]))
            info, key_ids = loaders.SecureIOInspectCeio().inspect(path)
            self.assertEqual(key_ids.split("\n"), ids)
            self.assertIn("no matching key on this host", info)
            self.assertIn("recovery.ceio.json", info)
        finally:
            os.environ.pop("SECUREIO_RECOVERY_KEY", None)

    def test_v1_when_single_recipient(self):
        path = self.seal(self.pub, "single")
        self.assertEqual(ceio.read_header(Path(path).read_bytes())["version"], 1)

    def test_delete_leaf_after_use(self):
        """Default on: the leaf and its cert are gone after a successful decrypt; the recovery key never is."""
        pub = make_leaf(self.ca, self.keys, "wipe1")
        path = self.seal(pub, "wipe1")
        out = loaders.LoadEncryptedImageWithMetadata().execute("", loaders.KEY_MATCH, ceio_path=path)
        self.assertIn("leaf deleted", out[15])
        self.assertFalse([f for f in os.listdir(self.keys) if f.startswith("wipe1")])
        with self.assertRaisesRegex(ceio.CeioError, "no key in"):
            run_loader("", loaders.KEY_MATCH, ceio_path=path)
        rec_dir = tempfile.mkdtemp(dir=SCRATCH)
        os.environ["SECUREIO_RECOVERY_KEY"] = os.path.join(rec_dir, "recovery.ceio.json")
        try:
            rec = make_leaf(self.ca, rec_dir, "recovery")
            leaf = make_leaf(self.ca, self.keys, "wipe2")
            p2 = nodes.SaveEncryptedImageWithFilename().save(self.img, leaf, "00_test", "sec_rec", "wipe2",
                                                             "overwrite", True, recovery_recipient=rec)["result"][0]
            out = loaders.LoadEncryptedImageWithMetadata().execute("", loaders.KEY_RECOVERY, ceio_path=p2)
            self.assertIn("kept", out[15])
            self.assertTrue(os.path.isfile(os.environ["SECUREIO_RECOVERY_KEY"]))
            self.assertTrue(os.path.isfile(os.path.join(self.keys, "wipe2.ceio.json")))   # unused leaf stays
        finally:
            os.environ.pop("SECUREIO_RECOVERY_KEY", None)

    def test_unverifiable_keys(self):
        nocert = make_leaf(self.ca, self.keys, "nocert", cert=False)
        with self.assertRaisesRegex(ceio.CeioError, "no cert next to"):
            run_loader("", loaders.KEY_MATCH,
                                                             ceio_path=self.seal(nocert, "nocert"))
        rogue_ca = make_ca(tempfile.mkdtemp(dir=SCRATCH))
        rogue = make_leaf(rogue_ca, self.keys, "rogue")
        with self.assertRaisesRegex(ceio.CeioError, "does not chain"):
            run_loader("", loaders.KEY_MATCH,
                                                             ceio_path=self.seal(rogue, "rogue"))

    def test_refusals(self):
        L = loaders.LoadEncryptedImageWithMetadata
        for key in ("../good.ceio.json", "C:/x.ceio.json", "good.json"):
            self.assertIsNot(L.VALIDATE_INPUTS(self.rel, key), True, key)
        self.assertIsNot(L.VALIDATE_INPUTS("../output/x/scan01.png.ceio", "good.ceio.json"), True)
        # placeholder dropdowns pass validation (a wired key_file may be issued in the same run) but not execution
        self.assertIs(L.VALIDATE_INPUTS(loaders.IMAGE_WIRED, loaders.KEY_WIRED), True)
        self.assertIs(L.VALIDATE_INPUTS(loaders.IMAGE_WIRED, loaders.KEY_MATCH), True)
        with self.assertRaisesRegex(ValueError, "no key selected"):
            run_loader(self.rel, loaders.KEY_WIRED)
        with self.assertRaisesRegex(ValueError, "no image selected"):
            run_loader(loaders.IMAGE_WIRED, "good.ceio.json")
        with self.assertRaisesRegex(ceio.CeioError, "sealed to key_id"):
            run_loader(self.rel, "other.ceio.json")
        for bad in ("C:/Windows/x.png.ceio", os.path.join(self.inp, self.rel) + "\nsecond.ceio",
                    os.path.join(self.out, "x", "nope.png")):
            with self.assertRaises(ValueError, msg=bad):
                run_loader("", "", ceio_path=bad, key_file="good.ceio.json")

    def test_wired_paths(self):
        """ceio_path (an output-folder file, like a saver's saved_path) and key_file override the dropdowns."""
        res = nodes.SaveEncryptedImageWithFilename().save(self.img, self.pub, "00_test", "sec_e2e", "wired",
                                                          "overwrite", True, original="a/b.jpg",
                                                          original_sha256="cd" * 32)
        out = run_loader(
            "", loaders.KEY_WIRED, ceio_path=res["result"][0], key_file=os.path.join(self.keys, "good.ceio.json"))
        self.assertEqual((out[9], out[10], out[12]), ("00_test/sec_e2e", "wired", "a/b.jpg"))

    def test_require_original(self):
        with self.assertRaisesRegex(ValueError, "no library original"):
            nodes.SaveEncryptedImageWithFilename().save(self.img, self.pub, "x", "", "noorig", "overwrite", True,
                                                        original="", original_sha256="", require_original=True)
        self.assertFalse(os.path.exists(os.path.join(self.out, "x", "noorig.png.ceio")))
        res = nodes.SaveEncryptedImageWithFilename().save(self.img, self.pub, "x", "", "nullorig", "overwrite",
                                                          True, original="", original_sha256="")
        meta, _ = ceio.open_(Path(res["result"][0]).read_bytes(),
                             ceio._load_key(os.path.join(self.keys, "good.ceio.json")))
        self.assertEqual((meta["restore"]["schema"], meta["restore"]["original"]), (1, None))


class IssueTestLeafTest(unittest.TestCase):
    """Scratch test root -> SecureIOIssueTestLeaf -> saver -> loader, all through the node classes."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(dir=SCRATCH)
        cls.out, cls.inp, cls.keys, cls.ca = (os.path.join(cls.root, d) for d in ("output", "input", "keys", "ca"))
        for d in (cls.out, cls.inp, cls.keys):
            os.makedirs(d)
        folder_paths.set_output_directory(cls.out)
        folder_paths.set_input_directory(cls.inp)
        os.environ["SECUREIO_KEY_DIR"] = cls.keys
        os.environ["SECUREIO_TEST_ROOT"] = cls.ca
        os.environ.pop("SECUREIO_ROOT_CERT", None)
        make_ca(cls.ca)

    def test_issue_seal_open(self):
        keys_mod = sys.modules[PACK + ".keys"]
        key_file, cert_path, recipient, key_id, info = keys_mod.SecureIOIssueTestLeaf().issue("e2e")
        self.assertTrue(os.path.isfile(os.path.join(self.keys, key_file)) and os.path.isfile(cert_path))
        self.assertIn("chain verified", info)
        img = torch.rand(1, 24, 32, 3, generator=torch.Generator().manual_seed(3))
        res = nodes.SaveEncryptedImageWithFilename().save(img, recipient, "00_test", "sec_e2e", "leaf",
                                                          "overwrite", True)
        self.assertEqual(ceio.read_header(Path(res["result"][0]).read_bytes())["key_id"], key_id)
        out = run_loader("", "", ceio_path=res["result"][0], key_file=key_file)
        self.assertEqual(out[14], key_id)
        with self.assertRaises(ValueError):
            keys_mod.SecureIOIssueTestLeaf().issue("../x")

    def test_recovery_node(self):
        keys_mod = sys.modules[PACK + ".keys"]
        r1, k1, info1 = keys_mod.SecureIOTestRecoveryKey().get()
        r2, k2, info2 = keys_mod.SecureIOTestRecoveryKey().get()
        self.assertEqual((r1, k1), (r2, k2))
        self.assertIn("created now", info1)
        self.assertIn("existing", info2)
        self.assertIn("private key not touched", info2)
        self.assertTrue(os.path.isfile(os.path.join(self.ca, "recovery.crt")))
        _kf, _crt, leaf_rec, _kid, _info = keys_mod.SecureIOIssueTestLeaf().issue("rec")
        img = torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(5))
        path = nodes.SaveEncryptedImageWithFilename().save(img, leaf_rec, "00_test", "sec_rec", "node", "overwrite",
                                                           True, recovery_recipient=r1)["result"][0]
        out = run_loader("", loaders.KEY_RECOVERY, ceio_path=path)
        self.assertEqual(out[14], k1)


class FakeS3:
    """Just enough of boto3's S3 client for the B2 nodes."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, _name):
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for (b, k) in sorted(s3.objects) if b == Bucket and k.startswith(Prefix)]}
        return P()


class BatchRoundTripTest(unittest.TestCase):
    """W1 encrypt folder (leaf per image + recovery) -> W2 decrypt with leaves, re-seal to recovery only, upload
    -> W3 download, decrypt with recovery, compare pixels. B2 is an in-memory fake."""

    def test_round_trip(self):
        batch = sys.modules[PACK + ".batch"]
        keys_mod = sys.modules[PACK + ".keys"]
        root = tempfile.mkdtemp(dir=SCRATCH)
        out, inp, kdir, ca = (os.path.join(root, d) for d in ("output", "input", "keys", "ca"))
        for d in (out, inp, kdir):
            os.makedirs(d)
        folder_paths.set_output_directory(out)
        folder_paths.set_input_directory(inp)
        os.environ.update(SECUREIO_KEY_DIR=kdir, SECUREIO_TEST_ROOT=ca)
        for v in ("SECUREIO_ROOT_CERT", "SECUREIO_RECOVERY_KEY"):
            os.environ.pop(v, None)
        make_ca(ca)
        os.makedirs(os.path.join(inp, "album", "sub"))
        g = torch.Generator().manual_seed(7)
        want = {}
        for rel, fmt in (("album/a.png", "PNG"), ("album/b.jpg", "JPEG"), ("album/sub/c.png", "PNG")):
            arr = (torch.rand(20 + len(rel), 30, 3, generator=g).numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(inp, rel), format=fmt)
            want[os.path.splitext(os.path.basename(rel))[0]] = np.asarray(Image.open(os.path.join(inp, rel)).convert("RGB"))
        originals = {p: Path(inp, p).read_bytes() for p in ("album/a.png", "album/b.jpg", "album/sub/c.png")}

        rec_pub, rec_id, _ = keys_mod.SecureIOTestRecoveryKey().get()
        # W1 ----------------------------------------------------------------
        r = batch.SecureIOEncryptFolder().encrypt("album", "*", True, "secure/encrypted_in", "overwrite", rec_pub)["result"]
        self.assertEqual(r[2], 3)
        enc = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(out, "secure", "encrypted_in")) for f in fs]
        self.assertTrue(all(f.endswith(".ceio") for f in enc) and len(enc) == 3, enc)
        self.assertEqual(len([f for f in os.listdir(kdir) if f.endswith(".ceio.json")]), 3)   # a leaf per image
        for f in enc:
            h = ceio.read_header(Path(f).read_bytes())
            self.assertEqual((h["version"], h["recipients"], h["key_ids"][1]), (2, 2, rec_id))
        self.assertEqual({p: Path(inp, p).read_bytes() for p in originals}, originals)      # originals untouched
        # W2 ----------------------------------------------------------------
        L = batch.LoadEncryptedImagesFromFolder()
        imgs, _m, _bn, stems, rels, *_rest = L.execute("output", "secure/encrypted_in", True, loaders.KEY_MATCH, True)
        self.assertEqual(sorted(stems), ["a", "b", "c"])
        self.assertEqual([f for f in os.listdir(kdir) if f.endswith(".ceio.json")], [])      # leaves deleted
        fake = FakeS3()
        batch._b2 = lambda: fake
        up = batch.SecureIOUploadToB2()
        for img, stem, rel in zip(imgs, stems, rels):
            saved = nodes.SaveEncryptedImageWithFilename().save(img, rec_pub, "secure/vast_out", rel, stem,
                                                                "overwrite", True)["result"][0]
            self.assertEqual(ceio.read_header(Path(saved).read_bytes())["key_ids"], [rec_id])  # recovery only
            up.upload(saved, "bkt", "secureio-test/vast_out", "secure/vast_out")
        self.assertEqual(sorted(k for _, k in fake.objects), [      # worker output is re-encoded as PNG
            "secureio-test/vast_out/album/a.png.ceio", "secureio-test/vast_out/album/b.png.ceio",
            "secureio-test/vast_out/album/sub/c.png.ceio"])
        with self.assertRaisesRegex(ValueError, "only .ceio"):
            up.upload(os.path.join(inp, "album", "a.png"), "bkt", "secureio-test/vast_out", "secure/vast_out")
        fake.objects[("bkt", "secureio-test/vast_out/evil.ceio")] = b"not ciphertext at all, plaintext" * 4
        # W3 ----------------------------------------------------------------
        with self.assertRaisesRegex(ValueError, "isn't ciphertext"):
            batch.SecureIODownloadFromB2().download("bkt", "secureio-test/vast_out", "secure/from_b2", "overwrite")
        del fake.objects[("bkt", "secureio-test/vast_out/evil.ceio")]
        stage, n, _ = batch.SecureIODownloadFromB2().download("bkt", "secureio-test/vast_out", "secure/from_b2",
                                                              "overwrite")["result"]
        self.assertEqual(n, 3)
        imgs3, _m3, _bn3, stems3, rels3, *_r = L.execute("output", stage, True, loaders.KEY_RECOVERY, True)
        for img, stem in zip(imgs3, stems3):
            got = np.round(img[0].numpy() * 255).astype(np.uint8)
            self.assertTrue(np.array_equal(got, want[stem]), stem)
        self.assertTrue(os.path.isfile(os.path.join(ca, "recovery.ceio.json")))              # recovery kept
        plain = [os.path.join(dp, f) for dp, _, fs in os.walk(out) for f in fs if not f.endswith(".ceio")]
        self.assertEqual(plain, [])                                                           # no plaintext in output


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1], verbosity=2)
