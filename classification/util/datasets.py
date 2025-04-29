# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
import numpy as np

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD




def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    if args.dataset == 'CIFAR10':
        CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR10_STD = [0.2023, 0.1994, 0.2010]

    if is_train:
        transform = create_transform(
            input_size=32,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    return transform

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class CIFAR10CDataset(VisionDataset):
    mean = (0.4915, 0.4823, 0.4468)
    std = (0.2470, 0.2435, 0.2616)
    num_classes = 10
    image_size = 32

    filename = 'CIFAR-10-C.tar'
    base_folder = 'CIFAR-10-C'
    md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    per_severity = 10000

    severities = [1, 2, 3, 4, 5]
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
        'snow', 'frost', 'fog', 'spatter',
        'brightness', 'contrast', 'saturate',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    def __init__(self, root, download=False, extract_only=False,
                 severity=1, corruption='gaussian_noise',
                 transform=None, target_transform=None):
        assert severity in self.severities
        assert corruption in self.corruptions

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.slice = slice((severity - 1) * self.per_severity, severity * self.per_severity)

        if extract_only:
            if not os.path.exists(os.path.join(root, self.base_folder)):
                print(f'Extracting {self.__class__.__name__}')
                extract_archive(os.path.join(root, self.filename), root)

        # now load the picked numpy arrays
        images_file_path = os.path.join(self.root, self.base_folder, f'{corruption}.npy')
        self.data = np.load(images_file_path)[self.slice]
        labels_file_path = os.path.join(self.root, self.base_folder, f'labels.npy')
        self.targets = np.load(labels_file_path)[self.slice]

    def download(self):
        download_and_extract_archive(self.url, self.root, filename=self.base_folder, md5=self.md5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
