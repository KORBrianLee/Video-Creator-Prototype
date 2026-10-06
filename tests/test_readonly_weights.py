import os
from pathlib import Path
import tempfile
import unittest


class ReadonlyLoaderTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("torch and safetensors are needed")
        self.torch, self.save_file = torch, save_file
        parent = Path(os.environ["TEMP"]) / "readonly-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, ignore_cleanup_errors=True)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_same_values_as_the_library_loader_for_every_dtype(self):
        from safetensors.torch import load_file
        from local_video.readonly_weights import load_readonly
        torch = self.torch
        tensors = {"bf16": torch.randn(33, 17).to(torch.bfloat16), "f32": torch.randn(5, 3), "f16": torch.randn(4).half(),
                   "i64": torch.arange(7), "flag": torch.tensor([True, False]), "empty": torch.zeros(0, 4)}
        path = self.folder / "w.safetensors"
        self.save_file(tensors, str(path))
        mine, library = load_readonly(str(path)), load_file(str(path))
        self.assertEqual(set(mine), set(library))
        for name in tensors:
            self.assertEqual(mine[name].dtype, library[name].dtype, name)
            self.assertTrue(torch.equal(mine[name], library[name]), name)
        self.assertTrue(torch.equal(torch.nn.functional.linear(torch.ones(2, 17), mine["bf16"].float()),
                                    torch.nn.functional.linear(torch.ones(2, 17), library["bf16"].float())))

    def test_corrupt_headers_are_rejected(self):
        import json
        import struct
        from local_video.readonly_weights import load_readonly
        def write(header, payload=b"\0" * 16):
            raw = json.dumps(header).encode()
            path = self.folder / f"bad{len(list(self.folder.iterdir()))}.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
            return str(path)
        cases = [{"a": {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}},
                 {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}},
                 {"a": {"dtype": "X9", "shape": [1], "data_offsets": [0, 4]}},
                 {"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}]
        for header in cases:
            with self.assertRaises(ValueError, msg=header):
                load_readonly(write(header))

    def test_install_routes_library_loads_and_can_be_disabled(self):
        import safetensors.torch as library
        from local_video import readonly_weights
        original = library.load_file
        try:
            self.assertTrue(readonly_weights.install())
            self.assertIs(library.load_file, readonly_weights.load_readonly)
            os.environ["CVL_READONLY_WEIGHTS"] = "0"
            self.assertFalse(readonly_weights.enabled())
            path = self.folder / "x.safetensors"
            self.save_file({"a": self.torch.ones(3)}, str(path))
            self.assertTrue(self.torch.equal(readonly_weights.load_readonly(str(path))["a"], self.torch.ones(3)))
        finally:
            os.environ.pop("CVL_READONLY_WEIGHTS", None)
            library.load_file = original


if __name__ == "__main__":
    unittest.main()
