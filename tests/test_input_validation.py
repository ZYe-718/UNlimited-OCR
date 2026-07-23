import unittest
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

# Smart module mock loader for missing dependencies
class MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        mock = MagicMock()
        setattr(self, name, mock)
        return mock

class ModuleMockLoader:
    def __init__(self, mocked_prefixes):
        self.mocked_prefixes = mocked_prefixes

    def find_spec(self, fullname, path, target=None):
        for prefix in self.mocked_prefixes:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        mod = MockModule(spec.name)
        mod.__file__ = f"<mock {spec.name}>"
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass

import importlib.util
mock_prefixes = [
    "transformers",
    "torch",
    "torchvision",
    "addict",
    "numpy",
    "tqdm",
    "fitz",
    "einops",
    "easydict",
    "sglang",
    "flash_attn",
    "requests",
]
sys.meta_path.insert(0, ModuleMockLoader(mock_prefixes))

from modeling_unlimitedocr import load_image, load_pil_images
from PIL import Image


class TestLoadImageInputValidation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

        # Create a valid sample PNG image
        self.img_path = os.path.join(self.tmp_dir.name, "sample.png")
        img = Image.new("RGB", (50, 50), color="red")
        img.save(self.img_path)

        # Create a dummy PDF file path
        self.pdf_path = os.path.join(self.tmp_dir.name, "document.pdf")
        with open(self.pdf_path, "w") as f:
            f.write("%PDF-1.4 dummy pdf content")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_pdf_input_raises_value_error(self):
        """Passing a PDF file to load_image should raise a clear ValueError explaining the PDF workflow."""
        with self.assertRaises(ValueError) as ctx:
            load_image(self.pdf_path)

        err_msg = str(ctx.exception)
        self.assertIn("Unsupported input format: PDF", err_msg)
        self.assertIn("model.infer_multi()", err_msg)
        self.assertIn("pdf_to_images()", err_msg)

    def test_pdf_uppercase_extension_raises_value_error(self):
        """Case-insensitive check for .PDF extension."""
        uppercase_pdf = os.path.join(self.tmp_dir.name, "doc2.PDF")
        with open(uppercase_pdf, "w") as f:
            f.write("%PDF-1.4 dummy pdf content")

        with self.assertRaises(ValueError) as ctx:
            load_image(uppercase_pdf)

        err_msg = str(ctx.exception)
        self.assertIn("Unsupported input format: PDF", err_msg)

    def test_nonexistent_file_raises_file_not_found(self):
        """Passing a path that does not exist raises FileNotFoundError."""
        non_existent_path = os.path.join(self.tmp_dir.name, "does_not_exist.png")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_image(non_existent_path)

        self.assertIn("Image file not found", str(ctx.exception))

    def test_valid_image_loads_successfully(self):
        """Valid raster images should load cleanly as PIL Image objects."""
        loaded_img = load_image(self.img_path)
        self.assertIsNotNone(loaded_img)
        self.assertIsInstance(loaded_img, Image.Image)

    def test_load_pil_images_with_pdf_in_conversation(self):
        """Passing a PDF in conversations dictionary raises ValueError instead of AttributeError."""
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>Convert document to markdown",
                "images": [self.pdf_path],
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            load_pil_images(conversation)

        self.assertIn("Unsupported input format: PDF", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
