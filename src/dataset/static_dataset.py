from typing import List
import argparse
import random
from sortedcontainers import SortedDict
import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

from .utils import make_dataset
import src.dataset.transform as transform
from src.dataset.aux_dataset import VSPWAuxiliaryData

class StandardData(Dataset):
    def __init__(self, args: argparse.Namespace,
                 transform: transform.Compose,
                 data_list_path: str,
                 class_list: List[int],
                 return_paths: bool,
                 train: bool = True):
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list, _ = self.load_filenames(args.data_root, data_list_path, class_list)
        self.transform = transform
        self.return_paths = return_paths

        if hasattr(args, 'aux_train_name') and train:
            self.use_aux = True
            self.aux_train_data = VSPWAuxiliaryData(transform=transform, args=args)
        else:
            self.use_aux = False

    def load_filenames(self, data_root, data_list_path,class_list):
        return make_dataset(data_root, data_list_path, class_list)

    def __len__(self):
        return len(self.data_list)

    def reset_indices(self):
        self.aux_train_data.reset_indices()

    def __getitem__(self, index):

        label_class = []
        image_path, label_path = self.data_list[index%len(self.data_list)]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        undesired_class = []
        for c in label_class:
            if c in self.class_list:
                new_label_class.append(c)
            else:
                undesired_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        new_label = np.zeros_like(label)  # background
        for lab in label_class:
            indexes = np.where(label == lab)
            new_label[indexes[0], indexes[1]] = self.class_list.index(lab) + 1  # Add 1 because class 0 is for bg
        for lab in undesired_class:
            indexes = np.where(label == lab)
            new_label[indexes[0], indexes[1]] = 255

        ignore_pix = np.where(new_label == 255)
        new_label[ignore_pix[0], ignore_pix[1]] = 255

        if self.transform is not None:
            image, new_label = self.transform(image, new_label)

        if self.use_aux:
            aux_images, aux_labels, _ = self.aux_train_data.__getitem__(index)
            image = {'aux_images': aux_images, 'images': image}
            new_label = {'aux_labels': aux_labels, 'labels': new_label}

        if self.return_paths:
            return image, new_label, image_path, label_path
        else:
            return image, new_label, self.use_aux


class EpisodicData(Dataset):
    def __init__(self,
                 transform: transform.Compose,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace):

        self.shot = args.shot
        self.random_shot = args.random_shot
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list, self.sub_class_file_list = self.load_filenames(args.data_root, data_list_path, self.class_list)
        self.transform = transform

    def load_filenames(self, data_root, data_list_path, class_list):
        return make_dataset(data_root, data_list_path, class_list)

    def __len__(self):
        return len(self.data_list)

    def _build_support(self, image_path, label_path, class_chosen):
        """
        Build Support Set
        Args:
            image_path: Qry image path
            label_path: Qry label path
            class_chosen: class id of class in query
        """
        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        # == First, randomly choose indexes of support images  =
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        if self.random_shot:
            shot = random.randint(1, self.shot)
        else:
            shot = self.shot

        for k in range(shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = [self.class_list.index(class_chosen) + 1]

        # == Second, read support images and masks  ============
        for k in range(shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == shot and len(support_image_list) == shot

        return support_image_list, support_label_list, support_image_path_list, subcls_list, shot

    def __getitem__(self, index):

        # ========= Read query image + Chose class =========================
        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        for c in label_class:
            if c in self.class_list:  # current list of classes to try
                new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        # == From classes in query image, chose one randomly ===

        class_chosen = np.random.choice(label_class)
        new_label = np.zeros_like(label)
        ignore_pix = np.where(label == 255)
        target_pix = np.where(label == class_chosen)
        new_label[ignore_pix] = 255
        new_label[target_pix] = 1
        label = new_label

        # ========= Build support ==============================================
        support_image_list, support_label_list, support_image_path_list, \
                subcls_list, shot = self._build_support(image_path, label_path, class_chosen)

        # == Forward images through transforms =================
        if self.transform is not None:
            qry_img, target = self.transform(image, label)
            for k in range(shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])
                support_image_list[k] = support_image_list[k].unsqueeze(0)
                support_label_list[k] = support_label_list[k].unsqueeze(0)

        # == Reshape properly ==================================
        spprt_imgs = torch.cat(support_image_list, 0)
        spprt_labels = torch.cat(support_label_list, 0)

        return qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, [image_path]
