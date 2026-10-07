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


class ComposeKeyframeTests(unittest.TestCase):
    def test_kept_region_comes_from_the_start_frame(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            start, end, target = (Path(temp) / name for name in ("s.png", "e.png", "t.png"))
            Image.new("RGB", (200, 100), (255, 0, 0)).save(start)
            Image.new("RGB", (200, 100), (0, 0, 255)).save(end)
            frame_pick.compose_keyframe(start, end, [(0, 0, 0.4, 1)], target, feather=2)
            with Image.open(target) as image:
                self.assertEqual(image.getpixel((10, 50)), (255, 0, 0))
                self.assertEqual(image.getpixel((190, 50)), (0, 0, 255))


class PasteRegionTests(unittest.TestCase):
    def test_region_lands_at_the_requested_position(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as temp:
            source, base, target = (Path(temp) / name for name in ("s.png", "b.png", "t.png"))
            image = Image.new("RGB", (200, 100), (0, 0, 0))
            ImageDraw.Draw(image).rectangle((20, 20, 59, 79), fill=(255, 255, 0))
            image.save(source)
            Image.new("RGB", (200, 100), (0, 0, 255)).save(base)
            frame_pick.paste_region(source, (0.1, 0.2, 0.3, 0.8), base, (0.6, 0.2), target, feather=2)
            with Image.open(target) as result:
                self.assertEqual(result.getpixel((140, 50)), (255, 255, 0))
                self.assertEqual(result.getpixel((40, 50)), (0, 0, 255))

    def test_clean_plate_keeps_the_surroundings_out(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as temp:
            source, clean, base, target = (Path(temp) / n for n in ("s.png", "c.png", "b.png", "t.png"))
            Image.new("RGB", (200, 100), (0, 200, 0)).save(clean)
            image = Image.new("RGB", (200, 100), (0, 200, 0))
            ImageDraw.Draw(image).rectangle((30, 30, 49, 69), fill=(255, 255, 0))
            image.save(source)
            Image.new("RGB", (200, 100), (0, 0, 255)).save(base)
            frame_pick.paste_region(source, (0.1, 0.2, 0.35, 0.8), base, (0.6, 0.2), target, feather=2, clean=clean)
            with Image.open(target) as result:
                self.assertEqual(result.getpixel((145, 50)), (255, 255, 0))
                self.assertEqual(result.getpixel((124, 30)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
