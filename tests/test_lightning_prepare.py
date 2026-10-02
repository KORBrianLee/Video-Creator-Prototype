"""Derived GGUF scheduler tensor, preservation and source-integrity checks."""
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_lightning as prepare


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cvl-preparation-")
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, existing_alpha=False):
        name = b"model.diffusion_model.test"
        entries = struct.pack("<Q", len(name))+name+struct.pack("<IQIQ", 1, 4, 0, 0)
        payload = struct.pack("<4f", 1, 2, 3, 4)
        if existing_alpha:
            name = b"alphas_cumprod"
            entries += struct.pack("<Q", len(name))+name+struct.pack("<IQIQ", 1, 1000, 0, 32)
            payload += b"\0"*16+struct.pack("<1000f", *[0.9]*1000)
        header = b"GGUF"+struct.pack("<IQQ", 3, 2 if existing_alpha else 1, 0)+entries
        data_start = (len(header)+31)//32*32
        source = self.folder / "base.gguf"
        source.write_bytes(header+b"\0"*(data_start-len(header))+payload)
        return source, self.folder / "base-lightning-linear.gguf"

    def test_original_payload_preserved_and_global_alpha_tensor_added(self):
        source, destination = self.source()
        before = source.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        result = prepare.prepare(source, destination, digest)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(result["sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())
        self.assertTrue(result["preservation"]["original_tensor_payload_byte_identical"])
        original, prepared = prepare.inspect_gguf(source), prepare.inspect_gguf(destination)
        self.assertEqual(prepared["entries"][0]["raw"], original["entries"][0]["raw"])
        alpha = prepared["entries"][-1]
        self.assertEqual(alpha["name"], "alphas_cumprod")
        self.assertEqual(alpha["shape"], [1000])
        self.assertEqual(alpha["dtype"], 0)
        with destination.open("rb") as stream:
            stream.seek(prepared["data_start"]+alpha["offset"])
            values = struct.unpack("<1000f", stream.read(4000))
        self.assertEqual(list(values), prepare.linear_alphas())
        self.assertAlmostEqual(math.sqrt((1-values[-1])/values[-1]), 25.146114834, places=4)
        self.assertTrue(all(a > b > 0 for a, b in zip(values, values[1:])))

    def test_existing_global_alpha_replaced_without_duplicate_key(self):
        source, destination = self.source(existing_alpha=True)
        original = source.read_bytes()
        prepare.prepare(source, destination, hashlib.sha256(original).hexdigest())
        result = prepare.inspect_gguf(destination)
        self.assertEqual([e["name"] for e in result["entries"]].count("alphas_cumprod"), 1)
        self.assertEqual(result["tensor_count"], 2)
        self.assertEqual(source.read_bytes(), original)

    def test_prepared_model_reuse_requires_exact_hash(self):
        source, destination = self.source()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result = prepare.prepare(source, destination, digest)
        self.assertFalse(result["reused"])
        self.assertTrue(prepare.prepare(source, destination, digest)["reused"])
        destination.write_bytes(b"bad")
        self.assertFalse(prepare.prepare(source, destination, digest)["reused"])

    def test_wrong_source_hash_never_writes_derivative(self):
        source, destination = self.source()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare.prepare(source, destination, "0"*64)
        self.assertFalse(destination.exists())

    def test_original_and_unowned_target_cannot_be_overwritten(self):
        source, destination = self.source()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            prepare.prepare(source, source, digest)
        destination.write_bytes(b"unowned existing file")
        with self.assertRaises(FileExistsError):
            prepare.prepare(source, destination, digest)
        self.assertEqual(destination.read_bytes(), b"unowned existing file")


if __name__ == "__main__":
    unittest.main()
