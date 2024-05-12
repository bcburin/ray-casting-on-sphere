from PIL import Image
from numpy import array

from src.algorithms import ray_casting
from src.models import Scenario, LightSource, ColoredSphere

if __name__ == '__main__':
    img = Image.new("RGB", (800, 600), color="black")
    scenario = Scenario(
        observer_position=array([400, 300, -1000]),
        light_source=LightSource(position=array([400, -400, -400]), intensity=array([255, 255, 255])),
        ambient_light=array([15, 15, 15]),
        element=ColoredSphere(
            coef_diffusion=array([0.3, 0.3, 0.9]),
            coef_spectral=0.2,
            n_rugosity=5,
            center=array([400, 300, 50]),
            radius=200
        )
    )
    ray_casting(image=img, scenario=scenario, lambertian=False)
    img.save('sphere.png')
