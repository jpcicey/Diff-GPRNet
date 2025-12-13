# -*- coding: utf-8 -*-
# @Date     : 2025/12/13
# @Author   : Zhou
# @File     : MyDataset_d.py
# description : This is the dataset class for reading

import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_transforms(to_gray=True):
    """
    Construct image preprocessing pipeline.
    Note: ToTensor() automatically converts a PIL image with values [0,255] to a float tensor with values [0.0, 1.0].
    Therefore, no additional Normalize is needed, as the data is already in the [0,1] range.
    """
    transform_list = []

    if to_gray:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.ToTensor())  # Automatically /255 → [0, 1]

    return transforms.Compose(transform_list)

class MyDataset_d(Dataset):
    """
    Ground Penetrating Radar wavefield PNG dataset reader class.
    Used for reading paired samples of (full_wavefield, diff_wavefield).
    Images are converted to single-channel grayscale and normalized to [0, 1].
    """
    def __init__(self, input_dir: str, diff_dir: str, to_gray=True):
        self.input_dir = input_dir
        self.diff_dir = diff_dir

        self.transform = get_transforms(to_gray=to_gray)

        inputs = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])
        diffs = sorted([f for f in os.listdir(diff_dir) if f.lower().endswith('.png')])

        if inputs != diffs:
            only_in_input = set(inputs) - set(diffs)
            only_in_diff = set(diffs) - set(inputs)
            msg = "File name mismatch\n"
            if only_in_input:
                msg += f"Files only in the input directory: {sorted(only_in_input)}\n"
            if only_in_diff:
                msg += f"Files only in the diff directory: {sorted(only_in_diff)}\n"
            raise ValueError(msg)

        self.basenames = inputs
        print(f"Loaded {len(self.basenames)} pairs of PNG samples")
    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        fname = self.basenames[idx]
        input_path = os.path.join(self.input_dir, fname)
        diff_path = os.path.join(self.diff_dir, fname)

        inp = Image.open(input_path).convert('L')
        diff = Image.open(diff_path).convert('L')

        inp = self.transform(inp)
        diff = self.transform(diff)

        return inp, diff

if __name__ == "__main__":
    dataset = MyDataset_d(
        input_dir=r"D:\MyDataset\conv_data\full_cc",
        diff_dir=r"D:\MyDataset\conv_data\diff_cc",
        to_gray=True
    )

    print("Total number of datasets:", len(dataset))
    inp, diff = dataset[random.randint(0, len(dataset) - 1)]

    print(f"Input shape: {inp.shape}")
    print(f"Diff shape:  {diff.shape}")

    inp_np = inp.squeeze().numpy()
    diff_np = diff.squeeze().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax[0].imshow(inp_np, cmap='gray')
    ax[0].set_title('Input Image (Full Wavefield)')
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(diff_np, cmap='gray')
    ax[1].set_title('Diffraction Image')
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()