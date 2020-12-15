from torchvision.transforms import functional as F


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args
