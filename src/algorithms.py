import numpy as np
from PIL import Image
from numpy import array

from src.models import Scenario, ScenarioElement
from src.utils import normalize_vector, convert_array_to_rgb_tuple


def get_color_from_phong_model(
        scenario: Scenario, intersection_point: array, element: ScenarioElement, lambertian: bool = False):
    n = element.get_normal(point=intersection_point)
    l = normalize_vector(scenario.light_source.position - intersection_point)
    v = normalize_vector(scenario.observer_position - intersection_point)
    # point in shadow
    if element.point_is_in_shadow(light_source=scenario.light_source, point=intersection_point):
        return scenario.ambient_light
    # calculate diffusion coefficients
    diffusion = max(0, np.dot(n, l)) * element.coef_diffusion
    if lambertian:
        return diffusion * scenario.light_source.intensity + scenario.ambient_light
    # calculate spectral coefficients
    r = normalize_vector(2 * np.dot(n, l) * n - l)
    spectral = (np.dot(r, v) ** element.n_rugosity) * element.coef_spectral
    return (diffusion + spectral) * scenario.light_source.intensity + scenario.ambient_light


def ray_casting(image: Image, scenario: Scenario, lambertian: bool):
    width, height = image.size
    for x in range(width):
        for y in range(height):
            element = scenario.element
            intersection_point = element.get_intersection_point(
                observer_position=scenario.observer_position, pixel_position=array((x, y, 0)))
            if intersection_point is None:
                continue
            color = get_color_from_phong_model(
                scenario=scenario, element=element, intersection_point=intersection_point, lambertian=lambertian)
            image.putpixel((x, y), convert_array_to_rgb_tuple(v=color))
