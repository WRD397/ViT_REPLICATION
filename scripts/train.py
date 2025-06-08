import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))  # Adds root directory to sys.path

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.vit import VisionTransformerSmall
from utils.model_io import save_model
from utils.config_loader import load_config
from utils.data_loader import DatasetLoader
from pynvml import (
    nvmlInit, nvmlDeviceGetName, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for  inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Update progress bar with metrics
        if total > 0:
            avg_loss = running_loss / total
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{accuracy:.2f}%"
            })


    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Avoid division by zero on first step
            if total > 0:
                avg_loss = running_loss / total
                accuracy = 100. * correct / total

                progress_bar.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "Acc": f"{accuracy:.2f}%"
                })

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    # Load config
    config = load_config(f"{ROOT_DIR_PATH}/config/vit_config.yaml")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    print('loading training testing data')
    # loading cifar100
    cifar100_config = config["data"]['CIFAR100']
    DATASET = cifar100_config["dataset"]
    DATA_DIR = cifar100_config["data_path"]
    BATCH = cifar100_config["batch_size"]
    NUM_WORKERS = cifar100_config["num_workers"]
    IMAGE = cifar100_config["img_size"]

    # loading data
    print(f'loading dataset : {DATASET}')
    loader = DatasetLoader(dataset_name=DATASET,
                            data_dir=DATA_DIR,
                            batch_size=BATCH,
                            num_workers=NUM_WORKERS,
                            img_size=IMAGE)
    train_loader, val_loader = loader.get_loaders()
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print('data sanity check')
    for images, labels in train_loader:
        print(f'image shape and labels shape in training data - one batch : {images.shape}, {labels.shape}')
        break

    # Model
    modelConfig = config["model"]
    vitSmall_config = modelConfig['VIT_SMALL']
    MODEL_NAME = vitSmall_config["name"]
    LEARNING_RATE = vitSmall_config['lr']
    EPOCHS = vitSmall_config['epochs']

    model = VisionTransformerSmall(config).to(device)
    
    # printing model config
    print("Model Configuration:")
    for k, v in vitSmall_config.items():
        print(f"  {k}: {v}")
    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_model_state = None
    
    # gpu utilization
    max_mem_used = 0
    max_gpu_util = 0
    max_mem_util = 0

    # Training loop
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(handle)

    startTime = time.time()
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Monitor GPU usage
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util_info = nvmlDeviceGetUtilizationRates(handle)
        mem_used_mb = mem_info.used / (1024 ** 2)
        max_mem_used = max(max_mem_used, mem_used_mb)
        max_gpu_util = max(max_gpu_util, util_info.gpu)
        max_mem_util = max(max_mem_util, util_info.memory)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            corresponding_train_acc = train_acc
            corresponding_train_loss = train_loss
            best_model_state = model.state_dict()
    
  
    endTime = time.time()
    elapsedTime = endTime - startTime
    # Convert seconds to h:m:s
    hours = int(elapsedTime // 3600)
    minutes = int((elapsedTime % 3600) // 60)
    seconds = int(elapsedTime % 60)
    
    # Shutdown 
    nvmlShutdown()
    
    print('\n====== Model Performance =======')
    print(f"Train     Loss: {corresponding_train_loss:.4f},    Accuracy: {corresponding_train_acc:.2f}%")
    print(f"Val(Best) Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.2f}%")
    print('====== Hardware Performance =======')
    print(f"GPU Used : {gpu_name}")
    print(f"Peak GPU Memory: {max_mem_used:.2f} MB")
    print(f"Peak GPU Utilization: {max_gpu_util}%")
    print(f"Peak Memory Bandwidth Utilization: {max_mem_util}%\n")
    print(f"\nTraining completed in: {hours}h : {minutes}m : {seconds}s\n\n")

    # # Save model
    # save_model(model_state=best_model_state, model_name=modelName, epoch=epoch, val_acc=best_val_acc)
    # print(f"Model saved as {modelName} inside checkpoints")


if __name__ == "__main__":
    main()
