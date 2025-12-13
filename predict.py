# -*- coding: utf-8 -*-
# @Date     : 2025/12/13
# @Author   : Zhou
# @File     : predict.py
# description : This is the prediction code for Diff-GPRNet.

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diff_gprnet import UNet_RMA_Wavelet as UNet

class MinMaxNormalize:
    def __call__(self, array: np.ndarray) -> np.ndarray:
        amin = array.min()
        amax = array.max()
        if amax == amin:
            return np.zeros_like(array, dtype=np.float32)
        normalized = (array - amin) / (amax - amin)
        return normalized.astype(np.float32)

class ZScoreNormalize:
    def __call__(self, array: np.ndarray) -> np.ndarray:
        mean = array.mean()
        std = array.std()
        if std == 0:
            return np.zeros_like(array, dtype=np.float32)
        normalized = (array - mean) / std
        return normalized.astype(np.float32)

def read_png(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img).astype(np.float32)

def read_npy(path: str) -> np.ndarray:
    data = np.load(path)
    return data.astype(np.float32)

def read_csv(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=',')
    return data.astype(np.float32)

def auto_read_data(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        return read_png(path)
    elif ext == '.npy':
        return read_npy(path)
    elif ext == '.csv':
        return read_csv(path)

def predict_data(model_path, data_path, output_dir="pre_data", normalize=True):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded: {model_path}")

    data = auto_read_data(data_path)
    print(f"Input shape: {data.shape}")

    if normalize:
        norm = MinMaxNormalize()
        normalized_data = norm(data)
    else:
        normalized_data = data

    input_tensor = torch.from_numpy(normalized_data).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(input_tensor)

    pred_np = pred.squeeze().cpu().numpy().astype(np.float32)

    base_name = os.path.splitext(os.path.basename(data_path))[0]

    in_png = os.path.join(output_dir, f"{base_name}_input.png")
    in_csv = os.path.join(output_dir, f"{base_name}_input.csv")
    in_npy = os.path.join(output_dir, f"{base_name}_input.npy")
    
    out_png = os.path.join(output_dir, f"{base_name}_pred.png")
    out_csv = os.path.join(output_dir, f"{base_name}_pred.csv")
    out_npy = os.path.join(output_dir, f"{base_name}_pred.npy")

    plt.imsave(in_png, normalized_data, cmap="gray")
    np.savetxt(in_csv, normalized_data, fmt="%.6f", delimiter=",")
    np.save(in_npy, normalized_data)

    plt.imsave(out_png, pred_np, cmap="gray")
    np.savetxt(out_csv, pred_np, fmt="%.6f", delimiter=",")
    np.save(out_npy, pred_np)

    print(f"Results saved to {output_dir}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_data, cmap="gray", aspect='auto')
    plt.title("Input")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_np, cmap="gray", vmin=0, vmax=1, aspect='auto')
    plt.title("Prediction")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()