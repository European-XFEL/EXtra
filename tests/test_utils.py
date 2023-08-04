import numpy as np

from extra.utils import imshow2


def test_scaled_imshow():
    # Smoke test
    image = np.random.rand(100, 100)
    imshow2(image)
