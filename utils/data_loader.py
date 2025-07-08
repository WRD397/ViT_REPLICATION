import sys
import os
from dotenv import load_dotenv
load_dotenv()
ROOT_DIR_PATH = os.environ.get('ROOT_PATH')
sys.path.append(os.path.abspath(ROOT_DIR_PATH))  # Adds root directory to sys.path
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.config_loader import load_config
from torchvision.datasets import ImageFolder
from utils.tinyimagenet_setup import prepare_tiny_imagenet
from utils.caltech_setup import prepare_caltech256
from collections import defaultdict
import os
import shutil
from torch.utils.data import Subset
from collections import defaultdict
import random
from torch.utils.data import Dataset

# loading config file for CIFAR10
#config = load_config(f"{ROOT_DIR_PATH}/config/vit_test_config.yaml")
config = load_config(f"{ROOT_DIR_PATH}/config/vit_config.yaml")
data_cfg = config["data"]

class MappedSubset(Dataset):
    def __init__(self, dataset, indices, label_map):
        """
        dataset     : original ImageFolder dataset
        indices     : indices to keep
        label_map   : dict {original_label_index: new_label_index}
        """
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map  # maps original class indices to [0...N-1]

        # Remapped targets for quick lookup
        self.targets = [self.label_map[self.dataset[i][1]] for i in self.indices]

        # New class names and mapping
        used_orig_labels = sorted(self.label_map.keys())  # e.g., [3, 10, 17, 40]
        self.classes = [self.dataset.classes[orig] for orig in used_orig_labels]
        self.class_to_idx = {cls_name: new_idx for new_idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, original_label = self.dataset[self.indices[idx]]
        new_label = self.label_map[original_label]
        return image, new_label

class DatasetLoader:
    def __init__(self, training_config, dataset_name, data_dir, num_workers):
        self.dataset_name = dataset_name.upper()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.img_size = training_config['image_size']
        self.training_config = training_config
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if 'MNIST' in self.dataset_name
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    @staticmethod
    def _get_balanced_dataset(dataset, num_classes=None, samples_per_class=None, train=True, selected_classes=None):
        """
        For train=True:
            - randomly selects `num_classes`
            - selects `samples_per_class` images per class
        For train=False:
            - uses `selected_classes`
            - selects ALL images from each class
        """
        if not train : assert selected_classes is not None, "Validation needs selected_classes from training phase"
        else : assert (num_classes is not None ) & (samples_per_class is not None), "Training phase needs num_classes and samples_per_class"
        class_to_all_indices = defaultdict(list)

        # Build mapping from class_label to all indices
        for idx, (_, label) in enumerate(dataset):
            class_name = dataset.classes[label]
            class_to_all_indices[class_name].append(idx)

        if train:
            # Select num_classes randomly from available ones
            all_classes = list(class_to_all_indices.keys())
            selected_classes = random.sample(all_classes, num_classes)

        selected_indices = []
        label_map = {}
        print(f"{'[TRAINING]' if train else '[VALIDATION]'}")

        for new_label, class_name in enumerate(sorted(selected_classes)):
            indices = class_to_all_indices[class_name]
            orig_label = dataset.class_to_idx[class_name]
            #print(f"Class '{class_name}' (original label={orig_label}): total images = {len(indices)}", end='')

            if train:
                if len(indices) < samples_per_class:
                    raise ValueError(f"Class '{class_name}' has only {len(indices)} samples.")
                chosen = random.sample(indices, samples_per_class)
                #print(f", using = {samples_per_class}")
            else:
                chosen = indices  # all samples for validation
                #print(f", using = {len(chosen)}")
            selected_indices.extend(chosen)
            label_map[orig_label] = new_label
        print(f"Total selected samples: {len(selected_indices)}")
        return MappedSubset(dataset, selected_indices, label_map), selected_classes

    def get_dataset(self, train=True, training_classes = None):
        "Need to pass train_label_map when train = False ie. during validation"
        if not train : assert training_classes is not None, "Validation needs training classes" 
        TRAINING_CLASSES = training_classes
        AUG_ENABLED = self.training_config['augmentation_enabled']
        APPLY_CLASS_BALANCE = self.training_config['apply_class_balance']
        NUM_SUBSET_CLASS = self.training_config['num_subset_class']
        NUM_SUBSET_SAMPLE = self.training_config['num_subset_sample']

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
            if AUG_ENABLED:
                transform_cifar10 = train_transform_cifar10 if train else val_transform_cifar10
            else :
                transform_cifar10 = val_transform_cifar10
            
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
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar100, std_cifar100)
            ])
            
            if AUG_ENABLED:
                transform_cifar100 = train_transform_cifar100 if train else val_transform_cifar100
            else :
                transform_cifar100 = val_transform_cifar100
            return datasets.CIFAR100(root=self.data_dir, train=train, download=True, transform=transform_cifar100)
        elif self.dataset_name == 'MNIST':
            return datasets.MNIST(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'FASHIONMNIST':
            return datasets.FashionMNIST(root=self.data_dir, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'TINYIMAGENET200':
            tinyiimg_cfg = data_cfg['TINYIMAGENET200']
            mean_tinyimg= tinyiimg_cfg['mean_aug']
            std_tinyimg  = tinyiimg_cfg['std_aug']
            img_size = self.img_size
            train_transform_tinyimg = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                #transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean_tinyimg, std_tinyimg),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            ])

            val_transform_tinyimg = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_tinyimg, std_tinyimg)
            ])
            if AUG_ENABLED:
                transform_tinyimg = train_transform_tinyimg if train else val_transform_tinyimg
            else :
                transform_tinyimg = val_transform_tinyimg

            split_folder = "train" if train else "val"
            dataset_path = os.path.join(self.data_dir, "tiny-imagenet-200", split_folder)
            dataset = ImageFolder(root=dataset_path, transform=transform_tinyimg)
            if APPLY_CLASS_BALANCE:
                if train:
                    print(f'getting balanced subset - class count : {NUM_SUBSET_CLASS} - sample per class : {NUM_SUBSET_SAMPLE}')
                    train_dataset, selected_classes = self._get_balanced_dataset(
                                                                          dataset=dataset,train=train,
                                                                          num_classes=NUM_SUBSET_CLASS,
                                                                          samples_per_class=NUM_SUBSET_SAMPLE, selected_classes=TRAINING_CLASSES)
                    return train_dataset, selected_classes
                else :
                    val_dataset,_ = self._get_balanced_dataset(dataset=dataset,
                                                                            train=train,
                                                                            selected_classes=TRAINING_CLASSES)
                    return val_dataset
            else :
                return dataset
            
        elif self.dataset_name == 'CALTECH256':
            caltech_cfg = data_cfg['CALTECH256']
            mean_caltech = caltech_cfg['mean_aug']
            std_caltech = caltech_cfg['std_aug']
            train_transform_caltech = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean_caltech, std_caltech),
                transforms.RandomErasing(p=0.15, scale=(0.01, 0.05), ratio=(0.3, 3.3), value='random')
            ])
            val_transform_caltech = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean_caltech, std_caltech)
            ])

            transform_caltech = (
                train_transform_caltech if (train and AUG_ENABLED) else val_transform_caltech
            )

            split_folder = "train" if train else "val"
            clutter_dir = os.path.join(self.data_dir, split_folder, '257.clutter')
            if os.path.exists(clutter_dir):
                print("Removing clutter class...")
                shutil.rmtree(clutter_dir)
            dataset_path = os.path.join(self.data_dir, split_folder)
            dataset = ImageFolder(root=dataset_path, transform=transform_caltech)

            if APPLY_CLASS_BALANCE:
                if train:
                    print(f'getting balanced subset - class count : {NUM_SUBSET_CLASS} - sample per class : {NUM_SUBSET_SAMPLE}')
                    train_dataset, selected_classes = self._get_balanced_dataset(
                                                                          dataset=dataset,train=train,
                                                                          num_classes=NUM_SUBSET_CLASS,
                                                                          samples_per_class=NUM_SUBSET_SAMPLE, selected_classes=TRAINING_CLASSES)
                    return train_dataset, selected_classes
                else :
                    val_dataset,_ = self._get_balanced_dataset(dataset=dataset,
                                                                            train=train,
                                                                            selected_classes=TRAINING_CLASSES)
                    return val_dataset
                
            else:
                return dataset
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def get_loaders(self):
        if self.dataset_name == 'CALTECH256': prepare_caltech256()
        elif self.dataset_name == 'TINYIMAGENET200':prepare_tiny_imagenet()
        if self.training_config['apply_class_balance']:
            train_dataset, selected_classes = self.get_dataset(train=True)
            test_dataset = self.get_dataset(train=False, training_classes=selected_classes)
        else :
            train_dataset = self.get_dataset(train=True)
            test_dataset = self.get_dataset(train=False)

        BATCH_SIZE = self.training_config['batch_size']

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, pin_memory = True, num_workers=min(os.cpu_count(),self.num_workers), persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, pin_memory = True, num_workers=min(os.cpu_count(),self.num_workers), persistent_workers=True, prefetch_factor=4)
        
        print(f'training size  : {len(train_loader.dataset)}')
        print(f'validation size : {len(test_loader.dataset)}')
        if isinstance(train_dataset, Subset):
            subset_labels = [train_dataset.dataset[idx][1] for idx in train_dataset.indices]
            unique_labels = sorted(set(subset_labels))
            print(f"Subset contains {len(unique_labels)} unique classes")
            print(f"Sample label: {train_dataset.dataset[train_dataset.indices[0]][1]}")
        else:
            print(f"Classes: {len(train_dataset.classes)}")
            print(f"Sample label: {train_dataset[0][1]}")

        return train_loader, test_loader


if __name__ == "__main__":
    pass
    # sample code
    # loading cifar100
    # cifar100_config = data_cfg['CIFAR100']
    # DATASET = cifar100_config["dataset"]
    # DATA_DIR = cifar100_config["data_path"]
    # BATCH = cifar100_config["batch_size"]
    # NUM_WORKERS = cifar100_config["num_workers"]
    # IMAGE = cifar100_config["img_size"]

    # # loading data
    # loader = DatasetLoader(dataset_name=DATASET,
    #                         data_dir=DATA_DIR,
    #                         batch_size=BATCH,
    #                         num_workers=NUM_WORKERS,
    #                         img_size=IMAGE)
    # train_loader, test_loader = loader.get_loaders()
    # print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
