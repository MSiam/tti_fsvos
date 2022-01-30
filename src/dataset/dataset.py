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
from torch.utils.data.dataloader import default_collate

from src.dataset.temporal_dataset import TAOEpisodicTemporalData, VSPWEpisodicTemporalData
from src.dataset.static_dataset import StandardData, EpisodicData
from src.dataset.ytvos_dataset import YTVOSStandard, YTVOSEpisodic
from src.dataset.ytvos_transform import TrainTransform, TestTransform
from src.dataset.aux_dataset import YTVOSAuxiliaryData, VSPWAuxiliaryData
from src.dataset.nminivspw_dataset import NMiniVSPWStandardData, NMiniVSPWEpisodicData

def create_transform(args: argparse.Namespace):
    edge_aspratio = 'longest'
    if hasattr(args, 'edge_aspratio'):
        edge_aspratio = args.edge_aspratio

    aug_dic = {'randscale': transform.RandScale([args.scale_min, args.scale_max]),
               'randrotate': transform.RandRotate([args.rot_min, args.rot_max],
                                                  padding=[0 for x in args.mean],
                                                  ignore_label=255),
               'hor_flip': transform.RandomHorizontalFlip(),
               'vert_flip': transform.RandomVerticalFlip(),
               'crop': transform.Crop([args.image_size, args.image_size], crop_type='rand',
                                      padding=[0 for x in args.mean], ignore_label=255),
               'resize': transform.Resize(args.image_size, edge_aspratio=edge_aspratio)
               }

    train_transform = [aug_dic[name] for name in args.augmentations]
    train_transform += [transform.ToTensor(), transform.Normalize(mean=args.mean, std=args.std)]
    train_transform = transform.Compose(train_transform)
    return train_transform


def get_train_loader(args: argparse.Namespace,
                     return_paths: bool = False) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    assert args.train_split in [0, 1, 2, 3]
    collate_fn = default_collate
    if hasattr(args, 'pretrain_cl') and args.pretrain_cl:
        # This is pretraining with contrastive Learning and no labelled data
        if args.train_name == "ytvis":
            train_data = YTVOSAuxiliaryData(transform=TrainTransform(args.image_size), args=args,
                                            class_list=None) # It means to use train_split to create class_list
        else:
            train_transform = create_transform(args)
            train_data = VSPWAuxiliaryData(train_transform=transform, args=args)

    elif args.train_name == "ytvis":
        if hasattr(args, 'episodic_train') and args.episodic_train:
            train_data = YTVOSEpisodic(transform=TrainTransform(args.image_size),
                                       train=True, args=args)
        else:
            train_data = YTVOSStandard(transform=TrainTransform(args.image_size),
                                       train=True,
                                       args=args)

    elif args.train_name == "nminivspw":
        train_transform = create_transform(args)
        train_data = NMiniVSPWStandardData(args=args,
                                           transform=train_transform,
                                           class_list=None,
                                           return_paths=False,
                                           data_list_path=args.train_list,
                                           train=True)
    else:
        train_transform = create_transform(args)

        split_classes = get_split_classes(args)
        class_list = split_classes[args.train_name][args.train_split]['train']

        # ===================== Build loader =====================
        train_data = StandardData(transform=train_transform,
                                  class_list=class_list,
                                  return_paths=return_paths,
                                  data_list_path=args.train_list,
                                  args=args,
                                  train=True)

    if hasattr(args, 'aux_train_name') and not hasattr(args, 'pretrain_cl'):
        collate_fn = collate_aux_labelled_images
    elif hasattr(args, 'conditioned') and args.conditioned:
        collate_fn = collate_protos_labelled_images

    if args.distributed:
        world_size = torch.distributed.get_world_size()
        train_sampler = DistributedSampler(train_data) if args.distributed else None
        batch_size = int(args.batch_size / world_size) if args.distributed else args.batch_size
    else:
        train_sampler = None
        batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True,
                                               collate_fn=collate_fn)
    print('#############################', len(train_loader), ' ', batch_size, ' ', len(train_data))
    return train_loader, train_sampler

def collate_protos_labelled_images(batch):
    images = {'rgb': [], 'protos': []}
    labels = []
    protos = []

    for img, gt, aux_flag in batch:
        images['rgb'].append(img['rgb'])
        images['protos'].append(img['protos'])
        labels.append(gt)
    labels = torch.stack(labels)
    images['rgb'] = torch.stack(images['rgb'])
    return images, labels, [False]

def collate_aux_labelled_images(batch):
    aux_images = []
    aux_labels = []
    labelled_images = []
    labelled_gt = []

    for img, gt, _ in batch:
        aux_images.append(img['aux_images'])
        aux_labels.append(gt['aux_labels'])

        labelled_images.append(img['images'])
        labelled_gt.append(gt['labels'])

    aux_images = torch.stack(aux_images)
    images = torch.stack(labelled_images)
    gt = torch.stack(labelled_gt)
    aux_gt = torch.stack(aux_labels)

    return {"images": images, "aux_images": aux_images}, {'labels': gt, 'aux_labels': aux_gt}, 1


def get_val_loader(args: argparse.Namespace, split_type: str='val') -> torch.utils.data.DataLoader:
    """
        Build the episodic validation loader.
    """
    assert args.test_split in [0, 1, 2, 3, -1, 'default']

    if args.test_name == "default" and args.train_name == "ytvis":
        # Do not perform this in case of YTVIS dataset
        pass
    elif args.test_name == "default" and args.train_name == "nminivspw":
        val_transform = transform.Compose([
                transform.Resize(args.image_size),
                transform.ToTensor(),
                transform.Normalize(mean=args.mean, std=args.std)])
    else:
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
    if args.temporal_episodic_val > 0: ############# Episodic Temporal Datasets
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

        elif args.temporal_episodic_val == 3:
            val_transform = TestTransform(args.image_size)
            val_data = YTVOSEpisodic(transform=val_transform,
                                     train=False,
                                     args=args)

        elif args.temporal_episodic_val == 4:
            if split_type == 'val':
                list_path = args.val_list
            else:
                list_path = args.test_list
            args.split_type = split_type

            val_transform = TestTransform(args.image_size)
            val_data = NMiniVSPWEpisodicData(args=args,
                                             transform=val_transform,
                                             class_list=None,
                                             data_list_path=list_path)

        workers = 1 if args.workers > 0 else args.workers
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)

    elif args.episodic_val: ########## Episodic Static Datasets
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
    else: ############### Standard Datasets Static/Temporal
        if args.test_name == "default" and args.train_name == "ytvis":
            val_transform = TestTransform(args.image_size)
            val_data = YTVOSStandard(transform=val_transform,
                                     train=False,
                                     args=args)
        else:
            class_list = split_classes[args.train_name][args.train_split]['train']
            val_data = StandardData(args=args,
                                    transform=val_transform,
                                    class_list=class_list,
                                    return_paths=False,
                                    data_list_path=args.val_list,
                                    train=False)
        # TODO: Fix problem in distributed working with temporal data
#        val_sampler = DistributedSampler(val_data) if args.distributed else None
#        world_size = torch.distributed.get_world_size()
#        batch_size = int(args.batch_size_val / world_size) if args.distributed else args.batch_size_val
        val_sampler = None
        batch_size = args.batch_size_val

        workers = 1 if args.workers > 0 else args.workers

        collate_fn = default_collate
        if hasattr(args, 'conditioned') and args.conditioned:
            collate_fn = collate_protos_labelled_images

        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler,
                                                 collate_fn=collate_fn)

    return val_loader, val_transform
