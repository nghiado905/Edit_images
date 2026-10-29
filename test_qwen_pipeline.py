"""Check stage ordering, image references, and fail-fast boundaries without weights."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import model_downloads
import qwen_lightning_edit as runner


class QwenPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "crop.jpg"
        Image.new("RGB", (245, 128), "green").save(self.input)
        self.python = self.root / "python.exe"
        self.python.touch()

    def test_alignment_does_not_squeeze_upscaled_crop(self):
        source = self.root / "upscaled.png"
        Image.new("RGB", (980, 512), "green").save(source)
        image = runner.prepare_image(source, 1024)
        self.assertEqual(image.size, (992, 512))
        self.assertEqual(image.getpixel((991, 511)), (0, 128, 0))

    def test_exif_orientation_is_applied_before_resizing(self):
        source = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (128, 64), "blue").save(source, exif=exif)
        self.assertEqual(runner.prepare_image(source, 1024).size, (64, 128))

    def test_upscale_then_encode_then_edit_with_the_same_reference(self):
        events = []
        original_bytes = self.input.read_bytes()

        def upscale(source, destination, **kwargs):
            events.append("upscale")
            destination.mkdir()
            path = destination / "upscaled.png"
            Image.new("RGB", (980, 512), "green").save(path)
            return path

        def worker(stage, path):
            events.append(stage)
            job = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(job["images"]), 1)
            self.assertEqual((job["width"], job["height"]), (992, 512))
            self.assertEqual(job["steps"], 8)
            self.assertEqual(job["true_cfg_scale"], 1.0)
            with Image.open(job["images"][0]) as image:
                self.assertEqual(image.getpixel((0, 0)), (0, 128, 0))
            if stage == "encode":
                Path(job["embeddings"]).touch()
            else:
                self.assertTrue(Path(job["embeddings"]).is_file())
                Image.new("RGB", (job["width"], job["height"])).save(job["output"])

        argv = ["qwen_lightning_edit.py", "--input", str(self.input), "--output-dir", str(self.root / "out")]
        with patch.object(sys, "argv", argv), patch.object(runner, "PYTHON", self.python), patch.object(
            runner, "missing_models", return_value=[]
        ), patch.object(runner, "upscale_image", side_effect=upscale), patch.object(runner, "run_worker", side_effect=worker):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(events, ["upscale", "encode", "edit"])
        self.assertEqual(self.input.read_bytes(), original_bytes)

    def test_encoding_failure_never_starts_edit(self):
        argv = ["qwen_lightning_edit.py", "--input", str(self.input), "--skip-upscale",
                "--output-dir", str(self.root / "out")]
        with patch.object(sys, "argv", argv), patch.object(runner, "PYTHON", self.python), patch.object(
            runner, "missing_models", return_value=[]
        ), patch.object(runner, "upscale_image") as upscale, patch.object(
            runner, "run_worker", side_effect=subprocess.CalledProcessError(1, "encode")
        ) as worker:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main()
        upscale.assert_not_called()
        self.assertEqual([call.args[0] for call in worker.call_args_list], ["encode"])

    def test_release_file_hash_is_checked_before_rename(self):
        data = b"test model file"
        release = {"repo_id": "test/model", "revision": "abc", "files": [{
            "rfilename": "weights/model.bin", "size": len(data),
            "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
        }]}
        destination = self.root / "download"
        def download(url, path):
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        with patch.object(model_downloads, "curl_download", side_effect=download):
            model_downloads.download_release(release, destination)
        self.assertEqual((destination / "weights" / "model.bin").read_bytes(), data)
        self.assertFalse(list(destination.rglob("*.partial")))

    def test_release_cannot_write_outside_destination(self):
        release = {"repo_id": "test/model", "revision": "abc", "files": [{"rfilename": "../../outside.bin", "size": 1}]}
        with self.assertRaises(ValueError):
            model_downloads.download_release(release, self.root / "download")


if __name__ == "__main__":
    unittest.main()
