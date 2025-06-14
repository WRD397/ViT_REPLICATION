import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))  # Adds root directory to sys.path
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.config_loader import load_config
from torchvision.datasets import ImageFolder
from utils.tinyimagenet_setup import prepare_tiny_imagenet

# loading config file for CIFAR10
#config = load_config(f"{ROOT_DIR_PATH}/config/vit_test_config.yaml")
config = load_config(f"{ROOT_DIR_PATH}/config/vit_config.yaml")
data_cfg = config["data"]

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

    def get_dataset(self, train=True, transform=None):
        if self.dataset_name == 'CIFAR10':
            #data augmentation - tackle overfitting problem, specially on small datasets like cifar10
            # CIFAR-10 mean & std
            cifar10_cfg = data_cfg['CIFAR10']
            mean_cifar10 = cifar10_cfg['mean_aug']
            std_cifar10  = cifar10_cfg['std_aug']  
            img_size = self.img_size
            train_transform_cifar10 = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4), ## less harsh
                #transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandAugment(num_ops=2, magnitude=9), # alternate to color jitter
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar10, std_cifar10)
            ])
            val_transform_cifar10 = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar10, std_cifar10)
            ])
            transform_cifar10 = train_transform_cifar10 if train else val_transform_cifar10
            return datasets.CIFAR10(root=self.data_dir, train=train, download=True, transform=transform_cifar10)
        elif self.dataset_name == 'CIFAR100':
            cifar100_cfg = data_cfg['CIFAR100']
            mean_cifar100 = cifar100_cfg['mean_aug']
            std_cifar100  = cifar100_cfg['std_aug']
            img_size = self.img_size
            train_transform_cifar100 = transforms.Compose([
                #transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar100, std_cifar100),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            ])
            val_transform_cifar100 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar100, std_cifar100)
            ])
            transform_cifar100 = train_transform_cifar100 if train else val_transform_cifar100
            return datasets.CIFAR100(root=self.data_dir, train=train, download=True, transform=transform_cifar100)
        elif self.dataset_name == 'MNIST':
            return datasets.MNIST(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'FASHIONMNIST':
            return datasets.FashionMNIST(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'TINYIMAGENET':
            prepare_tiny_imagenet()
            tinyiimg_cfg = data_cfg['TINYIMAGENET']
            mean_tinyimg= tinyiimg_cfg['mean_aug']
            std_tinyimg  = tinyiimg_cfg['std_aug']
            subset_enabled = tinyiimg_cfg['subset_enabled']
            subset_size = tinyiimg_cfg['subset_size']
            img_size = self.img_size
            train_transform_tinyimg = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean_tinyimg, std_tinyimg),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            ])

            val_transform_tinyimg = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_tinyimg, std_tinyimg)
            ])
            transform_tinyimg = train_transform_tinyimg if train else val_transform_tinyimg
            split_folder = "train" if train else "val"
            dataset_path = os.path.join(self.data_dir, "tiny-imagenet-200", split_folder)
            dataset = ImageFolder(root=dataset_path, transform=transform_tinyimg)
            if subset_enabled:
                if train:
                    subset_size = subset_size
                    indices = torch.randperm(len(dataset))[:subset_size]
                    subset = torch.utils.data.Subset(dataset, indices)
                    subset.classes = dataset.classes
                    subset.class_to_idx = dataset.class_to_idx
                    subset.targets = [dataset.targets[i] for i in indices] 
                    dataset = subset           
            return dataset
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def get_loaders(self):
        train_dataset = self.get_dataset(train=True)
        test_dataset = self.get_dataset(train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, pin_memory = True, num_workers=min(os.cpu_count(),self.num_workers), persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, pin_memory = True, num_workers=min(os.cpu_count(),self.num_workers), persistent_workers=True, prefetch_factor=4)
        
        print(f'training size  : {len(train_loader.dataset)}')
        print(f'validation size : {len(test_loader.dataset)}')
        return train_loader, test_loader


if __name__ == "__main__":
    # sample code
    # loading cifar100
    cifar100_config = data_cfg['CIFAR100']
    DATASET = cifar100_config["dataset"]
    DATA_DIR = cifar100_config["data_path"]
    BATCH = cifar100_config["batch_size"]
    NUM_WORKERS = cifar100_config["num_workers"]
    IMAGE = cifar100_config["img_size"]

    # loading data
    loader = DatasetLoader(dataset_name=DATASET,
                            data_dir=DATA_DIR,
                            batch_size=BATCH,
                            num_workers=NUM_WORKERS,
                            img_size=IMAGE)
    train_loader, test_loader = loader.get_loaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
