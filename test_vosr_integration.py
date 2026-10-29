"""CPU-only checks for the VOSR/edit boundary; no model downloads required."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from test_gen import prepare_edit_image
import vosr_upscale


class VOSRBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "VOSR"
        checkpoint = self.repo / "preset" / "ckpts" / "VOSR2"
        checkpoint.mkdir(parents=True)
        (checkpoint / "args.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").touch()
        self.python = self.root / "python.exe"
        self.python.touch()
        self.input = self.root / "crop.jpg"
        Image.new("RGB", (245, 128), "green").save(self.input)

    def invoke(self):
        return vosr_upscale.upscale_image(self.input, self.root / "output", python=self.python)

    def test_subprocess_contract_and_exact_crop_dimensions(self):
        original_bytes = self.input.read_bytes()

        def generate(command, **kwargs):
            self.assertEqual(kwargs["cwd"], self.repo)
            self.assertEqual(kwargs["env"]["TORCHDYNAMO_DISABLE"], "1")
            self.assertEqual(kwargs["env"]["XFORMERS_DISABLED"], "1")
            self.assertTrue(kwargs["check"])
            staged_input = Path(command[command.index("-i") + 1])
            destination = Path(command[command.index("-o") + 1])
            destination.mkdir()
            with Image.open(staged_input) as image:
                self.assertEqual(image.size, (248, 128))
                image.resize((992, 512)).save(destination / "input.png")

        with patch.object(vosr_upscale, "REPO", self.repo), patch.object(vosr_upscale.subprocess, "run", side_effect=generate):
            result = self.invoke()
        with Image.open(result) as image:
            self.assertEqual(image.size, (980, 512))
        self.assertEqual(self.input.read_bytes(), original_bytes)
        self.assertFalse(list(result.parent.glob("vosr-*")))

    def test_upstream_exit_zero_without_output_is_an_error(self):
        with patch.object(vosr_upscale, "REPO", self.repo), patch.object(vosr_upscale.subprocess, "run"):
            with self.assertRaisesRegex(RuntimeError, "produced no image"):
                self.invoke()

    def test_upstream_failure_does_not_fall_back_to_original(self):
        with patch.object(vosr_upscale, "REPO", self.repo), patch.object(
            vosr_upscale.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "vosr")
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                self.invoke()

    def test_edit_uses_rectangular_upscale_and_can_keep_full_resolution(self):
        upscaled = self.root / "upscaled.png"
        Image.new("RGB", (980, 512), "green").save(upscaled)
        image = prepare_edit_image(upscaled, 768)
        self.assertEqual(image.size, (768, 408))
        full = prepare_edit_image(upscaled, 0)
        self.assertEqual(full.size, (984, 512))


if __name__ == "__main__":
    unittest.main()
