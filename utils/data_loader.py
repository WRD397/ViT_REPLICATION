import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DatasetLoader:
    def __init__(self, dataset_name, data_dir,
                 batch_size, num_workers, img_size):
        self.dataset_name = dataset_name.upper()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if 'MNIST' in self.dataset_name
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_dataset(self, train=True):
        if self.dataset_name == 'CIFAR10':
            return datasets.CIFAR10(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'CIFAR100':
            return datasets.CIFAR100(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'MNIST':
            return datasets.MNIST(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'FASHIONMNIST':
            return datasets.FashionMNIST(root=self.data_dir, train=train, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def get_loaders(self):
        train_dataset = self.get_dataset(train=True)
        test_dataset = self.get_dataset(train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader


if __name__ == "__main__":
    import sys
    import os
    from utils.config_loader import load_config
    CURR_PATH = f'/home/wd/Documents/work_stuff/ViT_REPLICATION'
    sys.path.append(os.path.abspath(CURR_PATH))  # Adds root directory to sys.path

    # loading config file for CIFAR10
    config = load_config(f"{CURR_PATH}/config/vit_test_config.yaml")
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
    train_loader, test_loader = loader.get_loaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
