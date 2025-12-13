# -*- coding: utf-8 -*-
# @Date     : 2025/12/13
# @Author   : Zhou
# @File     : train.py
# description : This is the training code for Diff-GPRNet.

import os
import sys
import torch
import torch.nn as nn
import time
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from MyDataset_d import MyDataset_d

# from pytorch_msssim import ssim
from diff_gprnet import Diff_GPRNet as UNet
import logging
import matplotlib.pyplot as plt
import numpy as np

# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True      # Speed up!
#     torch.backends.cudnn.deterministic = False # Allow fastest algorithm
# ------------------------------------------
old_time = time.time()
# ------------------------------------------
# Set basic hyperparameters
batch_size = 24
lr = 3e-4
epochs = 210

temp_path = r'model_parameter/diff_gprnet_Psnr_temp.pth'
# temp_path = r'model_parameter/unet_rma_nodwt_h1_24_rehou_best.pth'
# Define the path to save the best model
best_path = r'model_parameter/diff_gprnet_Psnr_best.pth'
# Define the log file path
log_file_path = r'training_log_diff_gprnet_Psnr.txt'
# ------------------------------------------
dataset = MyDataset_d(
    input_dir=r"D:\MyDataset\conv_data\full_cc",
    diff_dir=r"D:\MyDataset\conv_data\diff_cc",
    # normalize=True,
    to_gray=True
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, 
                                           [train_size, test_size])    
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# print("Number of training samples:", len(train_dataset))
# print("Number of test samples:", len(test_dataset))
# ------------------------------------------

# Define PSNR calculation function
def calculate_psnr(img1, img2, data_range=1.0):
    """Calculate PSNR metric"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

# Function to create log file
def save_training_log(epoch, train_loss, eval_loss, eval_psnr, lr, file_path):
    """Save training information to txt file (SSIM removed)"""
    with open(file_path, 'a') as f:
        if epoch == 1:
            # If it's the first epoch, write the header first
            f.write("Epoch,Train_Loss,Eval_Loss,Eval_PSNR,Learning_Rate,Timestamp\n")
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{epoch},{train_loss:.6f},{eval_loss:.6f},{eval_psnr:.4f},{lr:.2e},{timestamp}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define network structure
    MyNet = UNet().to(device)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(MyNet.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with warm restarts scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,        # Restart every 30 epochs
        T_mult=2,      # Double the period each time
        eta_min=5e-7
    )

    # Loss function
    loss_fn = nn.HuberLoss().to(device)
    
    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # alpha = 0.1
    # Resume training from checkpoint
    start_epoch = 0
    best_eval = float('inf')
    best_psnr = 0.0
    # best_ssim = 0.0  # Commented out
    if os.path.isfile(temp_path):
        CONT = True
    else:
        CONT = False
        print("Model parameter file not found")
    if CONT:
        path_checkpoint = temp_path
        checkpoint = torch.load(path_checkpoint, map_location=device)
        MyNet.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        best_eval = checkpoint['best_eval']
        best_psnr = checkpoint.get('best_psnr', 0.0)
        # best_ssim = checkpoint.get('best_ssim', 0.0)
        print("Successfully loaded the saved model from interruption")

    train_losses = []
    eval_losses = []
    eval_psnrs = []  # Store PSNR metrics
    # eval_ssims = []

    # Initialize log file (if it doesn't exist)
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write("Epoch,Train_Loss,Eval_Loss,Eval_PSNR,Learning_Rate,Timestamp\n")
        print(f"Created log file: {log_file_path}")

    # Start trainingt training
    for epoch in range(start_epoch + 1, epochs + start_epoch + 1):
        # ===================Training Mode==================
        MyNet.train()
        train_loss = 0  # Initialize train_loss
        nan_count = 0   # Initialize nan_count
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, file=sys.stdout)
        for batch_idx, (inp, diff) in loop:  # Added batch_idx
            inp, diff = inp.to(device), diff.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast('cuda'):
                output = MyNet(inp)
                # Calculate loss
                loss = loss_fn(output, diff)
            # Backward pass and gradient update with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

            # Update progress bar
            loop.set_description(f'Epoch [{epoch}/{start_epoch + epochs}]') 
            loop.set_postfix(loss=loss.item())
        
        # Record training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===================Evaluation Mode==================
        MyNet.eval()
        eval_loss = 0
        eval_psnr = 0  # Initialize PSNR
        # eval_ssim = 0
        
        with torch.no_grad():
            for batch_idy, (inp, diff) in enumerate(test_loader):
                inp, diff = inp.to(device), diff.to(device)
                output = MyNet(inp)
                loss = loss_fn(output, diff)
                eval_loss += loss.item()
                
                # Calculate PSNR
                batch_psnr = calculate_psnr(output, diff)
                # batch_ssim = ssim(output, diff, data_range=1.0)
                
                eval_psnr += batch_psnr.item()
                # eval_ssim += batch_ssim.item()
        
        # Record evaluation loss and metrics
        avg_eval_loss = eval_loss / len(test_loader)
        avg_eval_psnr = eval_psnr / len(test_loader)
        # avg_eval_ssim = eval_ssim / len(test_loader)
        
        eval_losses.append(avg_eval_loss)
        eval_psnrs.append(avg_eval_psnr)
        # eval_ssims.append(avg_eval_ssim)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Save training log to txt file
        save_training_log(epoch, avg_train_loss, avg_eval_loss, avg_eval_psnr, current_lr, log_file_path)

        # Output training and validation loss every epoch
        if (epoch % 1 == 0) or (epoch == 1):
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Eval Loss: {avg_eval_loss:.6f}, "
                  f"PSNR: {avg_eval_psnr:.4f} dB, LR: {current_lr:.2e}")

        # Save best model - based on Eval Loss + PSNR
        improvement_threshold = 0.0
        current_score = (1.0 / avg_eval_loss) * 1.0 + avg_eval_psnr * 0  # Composite score (Loss-dominated only)
        
        if current_score > (1.0 / best_eval) * 1.0 + best_psnr * 0 + improvement_threshold:
            best_eval = avg_eval_loss
            best_psnr = avg_eval_psnr
            
            # Save the best model
            checkpoint = {
                'net': MyNet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_eval': best_eval,
                'best_psnr': best_psnr,
                # 'best_ssim': best_ssim,
            }
            torch.save(checkpoint, best_path)
            print(f"Epoch {epoch}: Validation metrics improved, saving best model to {best_path}")
            print(f"  Best metrics - Loss: {best_eval:.6f}, PSNR: {best_psnr:.4f} dB")

        # Save model parameters every 100 epochs
        if epoch % 100 == 0 and epoch != 0:
            checkpoint_dir = 'model_parameter'
            checkpoint_path = os.path.join(checkpoint_dir, f'model_parameter_{epoch}.pth')
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'net': MyNet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_eval': best_eval,
                'best_psnr': best_psnr,
                # 'best_ssim': best_ssim,
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Model saved: {checkpoint_path}")
        
        # Update learning rate
        scheduler.step()

new_time = time.time()
total_time = new_time - old_time

# Add summary information to the log file after training completion
with open(log_file_path, 'a', encoding='utf-8') as f:
    f.write(f"\nTraining Summary:\n")
    f.write(f"Total Epochs: {epochs}\n")
    f.write(f"Final Train Loss: {train_losses[-1]:.6f}\n")
    f.write(f"Final Eval Loss: {eval_losses[-1]:.6f}\n")
    f.write(f"Final PSNR: {eval_psnrs[-1]:.4f} dB\n")
    f.write(f"Best Eval Loss: {best_eval:.6f}\n")
    f.write(f"Best PSNR: {best_psnr:.4f} dB\n")
    f.write(f"Total Training Time: {total_time:.2f} seconds\n")
    f.write(f"Average Time per Epoch: {total_time/epochs:.2f} seconds\n")

print("Total runtime:", total_time, "seconds")
print(f"Training log saved to: {log_file_path}")
print(f"Best model saved to: {best_path}")

# ---------------Plotting----------------------
# Plot loss curves and PSNR curvesves and PSNR curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Loss plot
ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
ax1.plot(eval_losses, label='Eval Loss', color='red', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Evaluation Loss')
ax1.legend()
ax1.grid(True)

# PSNR plot
ax2.plot(eval_psnrs, label='PSNR (dB)', color='green', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('Evaluation PSNR')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_metrics_model_parameter.png')
plt.show()

# Plot PSNR trend separately
plt.figure(figsize=(10, 6))
plt.plot(eval_psnrs, label='PSNR (dB)', color='green', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('Evaluation PSNR Trend')
plt.legend()
plt.grid(True)
plt.savefig('metrics_trend_diff_gprnet_Psnr.png', dpi=300, bbox_inches='tight')
plt.show()