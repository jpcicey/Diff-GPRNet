# Diff-GPRNet 

Diff-GPRNet is an open-source deep learning framework for **diffraction wavefield reconstruction and separation in Ground Penetrating Radar (GPR) data**.

---

## Project Introduction


**Description:** Diffraction waves play a crucial role in characterizing small-scale subsurface heterogeneities. However, in practical GPR data, diffraction signals are often weak and heavily contaminated by reflections and noise. Diff-GPRNet addresses this challenge by learning diffraction-specific representations using a deep neural network with residual and attention mechanisms.

This repository provides a **complete workflow from data synthesis to model training and prediction**.

---

## Key Features

- Deep learning network for diffracted wavefield (Diff-GPRNet)
-  MATLAB-based synthesis of GPR wavefield data via time-domain convolution
- The code has a simple and lightweight structure, making it easy for beginners to deploy and modify.

---

## Repository Structure

- diff_gprnet.py : Definition of the Diff-GPRNet network structure
- modules/ : Network modules
- MyDataset_d.py : PyTorch dataset loader
- train.py : Model training script
- predict.py : Model prediction / inference script
- Dra_4.m : Main program for GPR wavefield synthesis (MATLAB)
- random_freq_xr_wrand.m : Frequency-dependent random perturbation function

---

## Project runtime environment
This project was developed and tested in the following environments:
- Win11/WSL2(Ubuntu 22.04)
- Python 3.10  
- PyTorch 2.1.2
