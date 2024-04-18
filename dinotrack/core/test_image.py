import pytest
from PIL import Image
from pathlib import Path
from typing import Union
from dinotrack.core.image import ReadImage


@pytest.mark.parametrize(
    "image, expected_result",
    [
        (["path/to/image1.jpg", "path/to/image2.jpg"], "expected_result1"),
        ("path/to/image3.jpg", "expected_result2"),
    ],
)
class TestReadImage:
    def test_call(self, image, expected_result):
        read_image = ReadImage()
        result = read_image(image)
        assert result == expected_result

    def test_image_open(self, image, expected_result):
        read_image = ReadImage()
        result = read_image(image)
        assert isinstance(result, Image.Image)

    def test_invalid_image(self, image, expected_result):
        read_image = ReadImage()
        with pytest.raises(Exception):
            read_image(image)

    def test_config(self, image, expected_result):
        config = {"model_name": "model", "width": 100, "height": 100}
        read_image = ReadImage(config)
        result = read_image(image)
        assert result == expected_result
