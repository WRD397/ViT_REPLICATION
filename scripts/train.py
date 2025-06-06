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
from model.vit import VisionTransformerTest
from utils.model_io import save_model
from utils.config_loader import load_config
from utils.data_loader import DatasetLoader

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
    config = load_config(f"{ROOT_DIR_PATH}/config/vit_test_config.yaml")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    print('loading training testing data')
    # loading config file for CIFAR10
    data_cfg = config["data"]
    DATASET = data_cfg["dataset"]
    DATA_DIR = data_cfg["data_path"]
    BATCH = data_cfg["batch_size"]
    NUM_WORKERS = data_cfg["num_workers"]
    IMAGE = data_cfg["img_size"]

    # loading data
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
    modelName = config["model"]["name"]
    model = VisionTransformerTest(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])

    best_val_acc = 0.0
    best_model_state = None
    
    # Training loop
    startTime = time.time()
    for epoch in range(config["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    endTime = time.time()
    elapsedTime = endTime - startTime
    # Convert seconds to h:m:s
    hours = int(elapsedTime // 3600)
    minutes = int((elapsedTime % 3600) // 60)
    seconds = int(elapsedTime % 60)

    print(f"Train     Loss: {train_loss:.4f},    Accuracy: {train_acc:.2f}%")
    print(f"Val(Best) Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.2f}%")
    print(f"\nTraining completed in: {hours}h : {minutes}m : {seconds}s")

    # # Save model
    # save_model(model_state=best_model_state, model_name=modelName, epoch=epoch, val_acc=best_val_acc)
    # print(f"Model saved as {modelName} inside checkpoints")


if __name__ == "__main__":
    main()
