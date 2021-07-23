import cv2
import numpy as np
import src.dataset.transform as transform
from .classes import get_split_classes, filter_classes
import torch
import random
import argparse
from typing import List
from torch.utils.data.distributed import DistributedSampler
import json
import os

from src.dataset.temporal_dataset import TAOEpisodicTemporalData, VSPWEpisodicTemporalData
from src.dataset.static_dataset import StandardData, EpisodicData


def get_train_loader(args: argparse.Namespace,
                     return_paths: bool = False) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    assert args.train_split in [0, 1, 2, 3]
    aug_dic = {'randscale': transform.RandScale([args.scale_min, args.scale_max]),
               'randrotate': transform.RandRotate([args.rot_min, args.rot_max],
                                                  padding=[0 for x in args.mean],
                                                  ignore_label=255),
               'hor_flip': transform.RandomHorizontalFlip(),
               'vert_flip': transform.RandomVerticalFlip(),
               'crop': transform.Crop([args.image_size, args.image_size], crop_type='rand',
                                      padding=[0 for x in args.mean], ignore_label=255),
               'resize': transform.Resize(args.image_size)
               }

    train_transform = [aug_dic[name] for name in args.augmentations]
    train_transform += [transform.ToTensor(), transform.Normalize(mean=args.mean, std=args.std)]
    train_transform = transform.Compose(train_transform)

    split_classes = get_split_classes(args)
    class_list = split_classes[args.train_name][args.train_split]['train']

    # ===================== Build loader =====================
    train_data = StandardData(transform=train_transform,
                              class_list=class_list,
                              return_paths=return_paths,
                              data_list_path=args.train_list,
                              args=args)

    world_size = torch.distributed.get_world_size()
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    batch_size = int(args.batch_size / world_size) if args.distributed else args.batch_size

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    return train_loader, train_sampler


def get_val_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the episodic validation loader.
    """
    assert args.test_split in [0, 1, 2, 3, -1, 'default']
    val_transform = transform.Compose([
            transform.Resize(args.image_size),
            transform.ToTensor(),
            transform.Normalize(mean=args.mean, std=args.std)])
    split_classes = get_split_classes(args)

    # ===================== Filter out classes seen during training =====================
    if args.test_name == 'default':
        test_name = args.train_name
        test_split = args.train_split
    else:
        test_name = args.test_name
        test_split = args.test_split
    class_list = filter_classes(args.train_name, args.train_split, test_name, test_split, split_classes)

    # ===================== Build loader =====================
    if args.temporal_episodic_val > 0:
        val_sampler = None
        if args.temporal_episodic_val == 1:
            val_data = TAOEpisodicTemporalData(transform=val_transform,
                                            class_list=class_list,
                                            data_list_path=args.val_list,
                                            args=args)
        elif args.temporal_episodic_val == 2:
            val_data = VSPWEpisodicTemporalData(transform=val_transform,
                                            class_list=class_list,
                                            data_list_path=args.val_list,
                                            args=args)

        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)

    elif args.episodic_val:
        val_sampler = None
        val_data = EpisodicData(transform=val_transform,
                                class_list=class_list,
                                data_list_path=args.val_list,
                                args=args)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)
    else:
        class_list = split_classes[args.train_name][args.train_split]['train']
        val_data = StandardData(args=args,
                                transform=val_transform,
                                class_list=class_list,
                                return_paths=False,
                                data_list_path=args.val_list)
        val_sampler = DistributedSampler(val_data) if args.distributed else None
        world_size = torch.distributed.get_world_size()
        batch_size = int(args.batch_size_val / world_size) if args.distributed else args.batch_size_val
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 sampler=val_sampler)

    return val_loader, val_transform
