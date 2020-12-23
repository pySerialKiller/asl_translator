from torchvision.transforms import functional as F
from torchvision import transforms as T


class Compsoe(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


class Normalize(T.Normalize):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args
