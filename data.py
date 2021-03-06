import numpy as np
import torch
import os
import PIL
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def open_image(path):
    img = PIL.Image.open(path)
    if img.mode  != 'L':
        img = img.convert('L')
    return img

class SegmentationDataset(Dataset):
    """Dataset suitable for segmentation tasks."""

    def __init__(self, image_dir, mask_dir, filenames, transform=None):
        """
        Args:
            root_dir (string): Directory with all images and masks.
            image_dir (string): The relative directory containing the images.
            mask_dir (string): The relative directory containing the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filenames = filenames
        self.mean = 0.
        self.std = 255.
        self.normalise = False
    
    def set_image_stats(self, mean, std):
        self.mean = mean
        self.std = std
    
    def set_normalisation(self, norm=True):
        self.normalise = norm

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.filenames[idx])
        image = np.asarray(open_image(img_name)).astype(np.float32)
        if self.normalise:
            image -= self.mean
            image /= self.std

        mask_name = os.path.join(self.mask_dir, self.filenames[idx])
        # If using categorical cross entropy, need an un-normalised long
        mask = np.asarray(open_image(mask_name)).astype(np.int_)
        sample = (image, mask)

        if self.transform:
            sample = self.transform(sample)

        return sample

class SegmentationBunch():
    """Associates batches of training, validation and testing datasets suitable
    for segmentation tasks."""
    def __init__(self, root_dir, image_dir, mask_dir, batch_size, valid_pct=0.1,
                 test_pct=0.0, transform=None):
        assert((valid_pct + test_pct) < 1.)
        image_dir = os.path.join(root_dir, image_dir)
        mask_dir = os.path.join(root_dir, mask_dir)
        transform = transform
        image_filenames = next(os.walk(image_dir))[2]
        random_list = np.random.choice(image_filenames, len(image_filenames),
                                       replace=False)
        valid_size = int(len(image_filenames) * valid_pct)
        test_size = int(len(image_filenames) * test_pct)
        train_size = len(image_filenames) - (valid_size + test_size)
        train_filenames = random_list[:train_size]
        valid_filenames = random_list[train_size:train_size + valid_size]
        
        train_ds = SegmentationDataset(image_dir, mask_dir, train_filenames,
            transform)
        valid_ds = SegmentationDataset(image_dir, mask_dir, valid_filenames,
            transform)
        
        mu = 0.0
        for img, _ in train_ds:
            mu += torch.mean(img)
        mu /= len(train_ds)
        var_diff = 0.0
        for img, _ in train_ds:
            var_diff += ((img - mu)**2).sum()
        N = len(train_ds) * np.prod(np.array(img.shape))
        std = np.sqrt(var_diff / (N - 1))

        self.mean = mu.item()
        self.std = std.item()
        train_ds.set_image_stats(*self.image_stats())
        train_ds.set_normalisation(True)
        self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        self.valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
        if test_size > 0:
            test_filenames = random_list[train_size + valid_size:]
            test_ds = SegmentationDataset(image_dir, mask_dir, test_filenames,
                transform)
            self.test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
        else:
            self.test_dl = None
    
    def image_stats(self):
        # This needs to be stored somewhere
        return 0.6969700455665588, 13.313282012939453
