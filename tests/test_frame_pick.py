from pathlib import Path
import json
import tempfile
import unittest

from local_video import frame_pick


class FakeControl:
    def __init__(self, root):
        self.root = root

    def normalize(self, value):
        scene = {"id": "s01", "seed": value["seed"], "negative_prompt": "default", **value["scenes"][0]}
        return {**value, "scenes": [scene], "_runtime_dir": str(self.root)}


class FakeNeo:
    def __init__(self):
        self.seen = []

    def settings_for(self, request):
        return {"width": 64, "height": 48}

    def first_frame(self, root, work, scene, settings, request, progress, is_cancelled):
        from PIL import Image
        self.seen.append(scene)
        Image.new("RGB", (64, 48), (scene["seed"] % 255, 0, 0)).save(work / "first-frame.png")
        return {"elapsed_seconds": 3.4}


class FramePickTests(unittest.TestCase):
    def test_one_sheet_and_small_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tmp").mkdir()
            neo = FakeNeo()
            summary = frame_pick.pick({"name": "f", "prompt": "an excavator", "seeds": [1, 2, 3, 4, 5, 6, 7]},
                                      FakeControl(root), neo, root / "reports" / "f")
            self.assertEqual([s["seed"] for s in neo.seen], [1, 2, 3, 4, 5, 6])
            self.assertEqual(neo.seen[0]["negative_prompt"], "default")
            self.assertTrue(Path(summary["sheet"]).is_file())
            self.assertEqual(len(summary["frames"]), 6)
            self.assertLess(len(json.dumps(summary)), 2048)

    def test_requires_seeds(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            frame_pick.pick({"prompt": "x", "seeds": []}, FakeControl(Path(temp)), FakeNeo(), Path(temp) / "r")


class ImportFramesTests(unittest.TestCase):
    def test_moves_and_crops_to_video_aspect_then_deletes_the_original(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            outside, root = Path(temp) / "outside", Path(temp) / "runtime"
            outside.mkdir()
            source = outside / "frame.jpg"
            Image.new("RGB", (1024, 576), (10, 20, 30)).save(source)
            result = frame_pick.import_frames([source], root, size=(160, 100))["imported"][0]
            self.assertFalse(source.exists())
            target = Path(result["path"])
            self.assertEqual(target.parent, root / "assets" / "first-frames")
            with Image.open(target) as image:
                self.assertEqual(image.size, (160, 100))


if __name__ == "__main__":
    unittest.main()
