import pytest
from PIL import Image
from pathlib import Path
from typing import Union

from dinotrack.core.image import ReadImage
from PIL import Image
import numpy as np


@pytest.fixture
def image():
    # Create a dummy black image
    width, height = 512, 512
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.mark.parametrize(
    "image",
    [
        ([image(), image()]),
        (image()),
    ],
)
class TestReadImage:
    def test_call(self, image):
        read_image = ReadImage()
        result = read_image(image)
        assert result

    def test_image_open(self, image):
        read_image = ReadImage()
        result = read_image(image)
        assert isinstance(result, Image.Image)

    def test_invalid_image(self, image):
        read_image = ReadImage()
        with pytest.raises(Exception):
            read_image("bad_image.jpg")

    def test_config(self, image, expected_result):
        config = {"model_name": "model", "width": 100, "height": 100}
        read_image = ReadImage(config)
        result = read_image(image)
        assert result == expected_result
