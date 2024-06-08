from torch.utils.data import Dataset
from PIL import ImageFilter
import random
import os
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
import torchvision.transforms as T
import numpy as np
from sklearn.model_selection import train_test_split


class CIFARAugmentations(object):
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            augmentations = [
                T.RandomResizedCrop(size=imgsize, scale=(min_scale_crops, max_scale_crops)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                T.RandomResizedCrop(size=imgsize, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                T.Resize(size=(int(imgsize[0]) + int(imgsize[0] * 0.1), int(imgsize[1]) + int(imgsize[1] * 0.1))),
                T.CenterCrop(size=imgsize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = T.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views)] if self.num_views > 1 else self.augmentations(x)


class ImageNetAugmentations(object):
    def __init__(self,
                 imgsize,
                 mean,
                 std,
                 mode="contrastive_pretrain",
                 jitter_strength=0.5,
                 min_scale_crops=0.2,
                 max_scale_crops=1.0,
                 ):
        if mode == "contrastive_pretrain":
            self.num_views = 2
            kernel_size = int(0.1 * imgsize[0])
            if kernel_size % 2 == 0:
                kernel_size += 1
            augmentations = [
                T.RandomResizedCrop(size=imgsize),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(
                    brightness=0.8 * jitter_strength,
                    contrast=0.8 * jitter_strength,
                    saturation=0.8 * jitter_strength,
                    hue=0.2 * jitter_strength
                )], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "train_classifier":
            self.num_views = 1
            augmentations = [
                T.RandomResizedCrop(size=imgsize),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        elif mode == "test_classifier":
            self.num_views = 1
            augmentations = [
                T.Resize(size=(int(imgsize[0]) + int(imgsize[0] * 0.1), int(imgsize[1]) + int(imgsize[1] * 0.1))),
                T.CenterCrop(size=imgsize),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        else: raise ValueError(f"Unrecognized mode: {mode}")

        self.augmentations = T.Compose(augmentations)

    def __call__(self, x):
        return [self.augmentations(x) for _ in range(self.num_views) ] if self.num_views > 1 else self.augmentations(x)


class GaussianBlur():
    def __init__(self, radius_min=0.1, radius_max=2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, x):
        return x.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


class CIFAR10Index(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(CIFAR10Index, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, targets = super().__getitem__(index)
        return images, targets, index


class STL10Index(STL10):
    def __init__(self, root, split="train", transform=None, target_transform=None, download=True):
        super(STL10Index, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, targets = super().__getitem__(index)
        return images, targets, index


class CIFAR100Index(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(CIFAR100Index, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        images, targets = super().__getitem__(index)
        return images, targets, index


class ImageNetIndex(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.imagefolder = ImageFolder(root, transform=transform)

    def __len__(self):
        return self.imagefolder.__len__()

    def __getitem__(self, index):
        images, targets = self.imagefolder.__getitem__(index)
        return images, targets, index


# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Subset
class SubsetIndex(Dataset):
    def __init__(self, dataset: Dataset,  indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        images, targets = self.dataset[self.indices[idx]]
        return images, targets, idx

    def __len__(self):
        return len(self.indices)


def build_dataset(
        dataset_name,
        train_transform_mode="contrastive_pretrain",
        val_transform_mode="test_classifier",
        test_transform_mode="test_classifier",
        val_split=0.0,
        random_state=42
):
    if dataset_name == 'cifar10':
        IMGSIZE = (32, 32)
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)
        NUM_CLASSES = 10
        if val_split == 0.0:
            train_dataset = CIFAR10Index(root='./data',download=True,train=True,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
            test_dataset = CIFAR10Index(root='./data',download=True, train=False,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
            val_dataset = test_dataset
            return train_dataset, val_dataset, test_dataset, NUM_CLASSES
        else:
            train_dataset = CIFAR10(root='./data',download=True,train=True, transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=train_transform_mode))
            val_dataset = CIFAR10(root='./data',download=True,train=True, transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=val_transform_mode))
            assert len(train_dataset) == len(val_dataset), "Train and val datasets have different lengths"
            train_idx, val_idx = train_test_split(
                np.arange(train_dataset.__len__()),
                test_size=val_split,
                shuffle=True,
                random_state=random_state,
                stratify=train_dataset.targets
            )
            # Subset-dataset for train and val
            train_dataset = SubsetIndex(train_dataset, train_idx)
            val_dataset = SubsetIndex(val_dataset, val_idx)
            # Indexed test-dataset
            test_dataset = CIFAR10Index(root='./data',download=True,train=False,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
            return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == 'cifar100':
        IMGSIZE = (32, 32)
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)
        NUM_CLASSES = 100
        if val_split == 0.0:
            train_dataset = CIFAR100Index(root='./data',download=True,train=True,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
            test_dataset = CIFAR100Index(root='./data',download=True, train=False,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
            val_dataset = test_dataset
            return train_dataset, val_dataset, test_dataset, NUM_CLASSES
        else:
            train_dataset = CIFAR100(root='./data',download=True,train=True, transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=train_transform_mode))
            val_dataset = CIFAR100(root='./data',download=True,train=True, transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=val_transform_mode))
            assert len(train_dataset) == len(val_dataset), "Train and val datasets have different lengths"
            train_idx, val_idx = train_test_split(
                np.arange(train_dataset.__len__()),
                test_size=val_split,
                shuffle=True,
                random_state=random_state,
                stratify=train_dataset.targets
            )
            # Subset dataset for train and val
            train_dataset = SubsetIndex(train_dataset, train_idx)
            val_dataset = SubsetIndex(val_dataset, val_idx)
            # Indexed test-dataset
            test_dataset = CIFAR100Index(root='./data',download=True,train=False,transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))

            return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == 'tinyimagenet':
        IMGSIZE = (64, 64)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        NUM_CLASSES = 200
        if val_split == 0.0:
            train_dataset = ImageNetIndex(root="./data/tiny-imagenet-200/train", transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
            test_dataset = ImageNetIndex(root="./data/tiny-imagenet-200/val/images", transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
            val_dataset = test_dataset
            return train_dataset, val_dataset, test_dataset, NUM_CLASSES
        else:
            train_dataset = ImageFolder(root="./data/tiny-imagenet-200/train", transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
            val_dataset = ImageFolder(root="./data/tiny-imagenet-200/train", transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=val_transform_mode))
            assert len(train_dataset) == len(val_dataset), "Train and val datasets have different lengths"
            train_idx, val_idx = train_test_split(
                np.arange(train_dataset.__len__()),
                test_size=val_split,
                shuffle=True,
                random_state=random_state,
                stratify=train_dataset.targets
            )
            # Subset dataset for train and val
            train_dataset = SubsetIndex(train_dataset, train_idx)
            val_dataset = SubsetIndex(val_dataset, val_idx)
            # Indexed test-dataset
            test_dataset = ImageNetIndex(root="./data/tiny-imagenet-200/val/images", transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))

            return train_dataset, val_dataset, test_dataset, NUM_CLASSES

    elif dataset_name == 'stl10':
        IMGSIZE = (96, 96)
        MEAN = (0.43, 0.42, 0.39)
        STD = (0.27, 0.26, 0.27)
        NUM_CLASSES = 10
        if val_split == 0.0:
            if "classifier" in train_transform_mode:
                train_dataset = STL10Index(root='./data',download=True,split="train",transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
                test_dataset = STL10Index(root='./data',download=True, split="test",transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
                val_dataset = test_dataset
            else:
                train_dataset = STL10Index(root='./data',download=True,split="train+unlabeled",transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=train_transform_mode))
                test_dataset = STL10Index(root='./data',download=True, split="test",transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))
                val_dataset = test_dataset
            return train_dataset, val_dataset, test_dataset, NUM_CLASSES

        else:
            if "classifier" in train_transform_mode:
                train_dataset = STL10(root='./data',download=True,split="train", transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=train_transform_mode))
                val_dataset = STL10(root='./data',download=True,split="train", transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=val_transform_mode))
                assert len(train_dataset) == len(val_dataset), "Train and val datasets have different lengths"
                train_idx, val_idx = train_test_split(
                    np.arange(train_dataset.__len__()),
                    test_size=val_split,
                    shuffle=True,
                    random_state=random_state,
                    stratify=train_dataset.labels
                )
                # Subset dataset for train and val
                train_dataset = SubsetIndex(train_dataset, train_idx)
                val_dataset = SubsetIndex(val_dataset, val_idx)
            else:
                train_dataset = STL10Index(root='./data',download=True,split="unlabeled", transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=train_transform_mode))
                val_dataset = STL10Index(root='./data',download=True,split="train", transform=CIFARAugmentations(imgsize=IMGSIZE, mean=MEAN, std=STD, mode=val_transform_mode))

            # Indexed test-dataset
            test_dataset = STL10Index(root='./data',download=True, split="test",transform=CIFARAugmentations(imgsize=IMGSIZE,mean=MEAN, std=STD, mode=test_transform_mode))

            return train_dataset, val_dataset, test_dataset, NUM_CLASSES

# For tinyImagenet
# https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py
# Download tiny-imagenet dataset 
# Run python data.py
def create_val_img_folder(data_dir, dataset):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    dataset_dir = os.path.join(data_dir, dataset)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


if __name__ == "__main__":
    create_val_img_folder("./data", "tiny-imagenet-200")
