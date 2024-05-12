from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils import normalize_vector


RGB = NDArray


@dataclass
class LightSource:
    position: NDArray
    intensity: RGB


@dataclass
class Scenario:
    light_source: LightSource
    ambient_light: RGB
    element: "ScenarioElement"
    observer_position: NDArray


@dataclass
class ScenarioElement(ABC):
    coef_diffusion: NDArray
    coef_spectral: float
    n_rugosity: int

    @abstractmethod
    def get_normal(self, point: NDArray) -> NDArray:
        ...

    @abstractmethod
    def get_intersection_point(self, observer_position: NDArray, pixel_position: NDArray) -> NDArray | None:
        ...

    @abstractmethod
    def point_is_in_shadow(self, light_source: LightSource, point: NDArray) -> bool:
        ...


@dataclass
class ColoredSphere(ScenarioElement):
    center: NDArray
    radius: float

    def get_normal(self, point: NDArray) -> NDArray:
        return normalize_vector(point - self.center)

    def get_intersection_point(self, observer_position: NDArray, pixel_position: NDArray) -> NDArray | None:
        direction = pixel_position - observer_position
        relative_position = observer_position - self.center
        # calculate second order equation coefficients
        a = np.dot(direction, direction)
        b = 2 * np.dot(direction, relative_position)
        c = np.dot(relative_position, relative_position) - self.radius**2
        # calculate second order equation determinant
        determinant = b*b - 4*a*c
        # no intersection
        if determinant < 0:
            return None
        # calculate solutions
        t1 = (-b - np.sqrt(determinant)) / (2*a)
        t2 = (-b + np.sqrt(determinant)) / (2*a)
        # return intersection considering occlusion
        if t1 >= 0 and t2 >= 0:
            return observer_position + min(t1, t2) * direction
        if t1 >= 0:
            return observer_position + t1 * direction
        if t2 >= 0:
            return observer_position + t2 * direction
        return None

    def point_is_in_shadow(self, light_source: LightSource, point: NDArray) -> bool:
        intersection_point = self.get_intersection_point(
            observer_position=light_source.position, pixel_position=point)
        return intersection_point is not None and np.linalg.norm(intersection_point - point) > 0.001
