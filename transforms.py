import torch
import numpy as np
from skimage import transform

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_categorical=False):
        self.is_categorical = is_categorical

    def __call__(self, sample):
        image, mask = sample

        image = torch.from_numpy(np.expand_dims(image, axis=0))
        if not self.is_categorical:
            mask = torch.from_numpy(np.expand_dims(mask, axis=0))
        else:
            mask = torch.from_numpy(mask)

        return (image, mask)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        mask = transform.resize(mask, (new_h, new_w))

        return (img, mask)
