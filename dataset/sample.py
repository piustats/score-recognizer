from typing import List, Dict

import torchvision.transforms.functional
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image


class sample(object):

    def __init__(self, image_path: str, annotations: List[Dict]):
        self.image_path = image_path
        self.img = Image.open(image_path)
        self.annotations = annotations
        self._aspect = self._calculate_aspect(self.img.size[0], self.img.size[1])

    def _calculate_aspect(self, width: int, height: int) -> str:
        def gcd(a, b):
            """The GCD (greatest common divisor) is the highest number that evenly divides both width and height."""
            return a if b == 0 else gcd(b, a % b)

        r = gcd(width, height)
        x = int(width / r)
        y = int(height / r)

        return f"{x}:{y}"

    def prepare(self, target_w: int, target_h: int):
        self._prepare_image(target_w, target_h)

    def _prepare_boxes(self):
        pass

    def _prepare_image(self, target_w: int, target_h: int):
        resize = T.Resize(size=(target_w, target_h))
        self.img = resize(self.img)
        self.img = T.functional.pil_to_tensor(self.img)
