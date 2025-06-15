import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))  # Adds root directory to sys.path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import time
from utils.checkpoints_manager import CheckpointManager

# -------------------------------
# 1. Initialize wandb
# -------------------------------
print('initializing wandb')
wandb.init(
    project="checkpoint_test", 
    name="mnist_cnn_run_epoch50_final",
    notes = "last chckpoints not getting deleted, resolved.",
    resume="never",
    config={
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "optimizer": "Adam"
})
wandb.define_metric("*", summary="none")  # suppress all

checkpoint_manager = CheckpointManager()

config = wandb.config

# -------------------------------
# 2. Define a simple CNN
# -------------------------------
print('defining the model')
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Output: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # Output: 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # Output: 64x7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# 3. Data Loaders
# -------------------------------
print('data loading')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not os.path.isdir(f"{ROOT_DIR_PATH}/data/MNIST"):
    os.makedirs(os.path.dirname(f"{ROOT_DIR_PATH}/data/MNIST"), exist_ok=True)

train_dataset = datasets.MNIST(root=f"{ROOT_DIR_PATH}/data/MNIST", train=True, transform=transform, download=True)
val_dataset   = datasets.MNIST(root=f"{ROOT_DIR_PATH}/data/MNIST", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=config.batch_size)

# -------------------------------
# 4. Setup training
# -------------------------------
print('setting up device and training')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

print('training starts...')
# -------------------------------
# 5. Training Loop
# -------------------------------
best_val_acc = 0.0
startTime = time.time()
for epoch in range(1,config.epochs+1):
    print(f"Epoch : {epoch}")
    # train
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    progress_bar = tqdm(train_loader, desc='training', leave=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix({
                    "TrainLoss": f"{running_loss:.4f}",
                    "TrainAcc": f"{correct/total:.2f}%"
                })



    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        progress_bar =tqdm(val_loader, desc='validation', leave=True)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                    "ValLoss": f"{val_loss:.4f}",
                    "ValAcc": f"{val_correct/val_total:.2f}%"
                })

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    import copy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        corresponding_train_acc = train_acc
        corresponding_train_loss = train_loss
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())


    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })
    # saving the latest best model/optimizer dict at 10 epochs interval
    checkpoint_manager.save_and_upload(
        epoch,
        best_model_state,
        best_optimizer_state,
        extra={"val_acc": val_acc}
    )


endTime = time.time()
elapsedTime = endTime - startTime
hours = int(elapsedTime // 3600)
minutes = int((elapsedTime % 3600) // 60)
seconds = int(elapsedTime % 60)
spent_time = f"{hours}h : {minutes}m : {seconds}s"
print(f"Elapsed Time : {spent_time}")

wandb.run.summary["performance"] = {
    "trainAcc": train_acc,
    "trainLoss": train_loss,
    "valLoss": val_loss,
    "valAcc": val_acc,
    "elapsedTime":spent_time
}


# finishing the run
wandb.finish()