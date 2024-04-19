from pathlib import Path
from typing import Union

import numpy as np
import pytest
from PIL import Image

from dinotrack.core.image import ReadImage

width, height = 512, 512
image_array = np.zeros((height, width, 3), dtype=np.uint8)
image = Image.fromarray(image_array)


@pytest.mark.parametrize(
    "image, count",
    [([image, image], 2), (image, 1), ("bad_path.jpg", 0)],
)
class TestReadImage:
    def test_call(self, image, count):
        inputs = None

        if isinstance(image, str) and not Path(image).exists():
            with pytest.raises(FileNotFoundError):
                inputs = ReadImage()(image)
        else:
            inputs = ReadImage()(image)

        if inputs is None:
            assert count == 0
        else:
            assert len(inputs["pixel_values"]) == count
