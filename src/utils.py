import numpy as np
from numpy.typing import NDArray


def normalize_vector(v: NDArray) -> NDArray:
    return v / np.linalg.norm(v)


def convert_array_to_rgb_tuple(v: NDArray) -> tuple[int, int, int]:
    return (
        int(v[0]) if v[0] <= 255 else 255,
        int(v[1]) if v[1] <= 255 else 255,
        int(v[2]) if v[2] <= 255 else 255,
    )
