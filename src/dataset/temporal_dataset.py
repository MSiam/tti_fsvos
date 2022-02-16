from typing import List
import argparse
import random
from sortedcontainers import SortedDict
import os
import json
import cv2
import numpy as np
import torch
import pickle

from src.dataset.static_dataset import EpisodicData
import src.dataset.transform as transform

class EpisodicTemporalData(EpisodicData):
    def __init__(self,
                 transform: transform.Compose,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace):
        self.root = args.data_root[1]
        args.data_root = args.data_root[0]

        super(EpisodicTemporalData, self).__init__(transform, class_list, data_list_path, args)

        with open(args.class_names_list, 'r') as f:
            self.external_classes = json.load(f)
            class_indices = np.array(class_list) - 1
            self.current_external_classes = list(np.array(self.external_classes)[class_indices])

        # Used to ensure constant support set for whole sequence
        self.current_sequence = None
        self.sequence_support = None
        self.class_chosen = None

    def _load_class_mapping(self, class_mapping_file):
        with open(class_mapping_file, "r") as f:
            class_mapping = json.load(f)
        return class_mapping

    def _create_list(self):
        pass

    def __getitem__(self, index):
        image_id = self.img_list[index]

        # ================== Load Query ========================
        image, image_path, pick_new_support = self._load_image(image_id)
        label_class, label = self._get_label(image_id, image.shape)

        if pick_new_support:
            self.class_chosen = np.random.choice(label_class)
        new_label = np.zeros_like(label)
        target_pix = np.where(label == self.class_chosen)
        new_label[target_pix] = 1
        label = new_label

        # ================== Build Support ========================
        if pick_new_support:
            support_image_list, support_label_list, support_image_path_list, \
                subcls_list, shot = self._build_support(None, None, self.class_chosen)

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

            self.sequence_support = (spprt_imgs, spprt_labels, subcls_list, support_image_path_list)
        else:
            if self.transform is not None:
                qry_img, target = self.transform(image, label)

            spprt_imgs, spprt_labels, subcls_list, support_image_path_list = self.sequence_support

        return qry_img, target, spprt_imgs, spprt_labels, subcls_list, \
                support_image_path_list, image_path

    def __len__(self):
        return len(self.img_list)

class VSPWEpisodicTemporalData(EpisodicTemporalData):
    def __init__(self,
                 transform: transform.Compose,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace):
        super(VSPWEpisodicTemporalData, self).__init__(transform, class_list, data_list_path, args)
        # Sequences with diff definition for Pascal Class, E.g. Bottle vs Bottle_or_Cup
        # Or Objects Completely Out of View
        self.ignore_seqs = ["1146_lps8_r-6J00", "118_fgmRMeHommU", "1278_C4zCNFn1xIs",
                            "1275_ARcg-EyKWrA", "1302_OCLlE02BHGk", "131_eUiWFntut00",
                            "1339_J73l0QCY8YM", "1353_CszoyQ3HMcM", "1476_75AL-XN84cI",
                            "1887_tUt0N6eGtGY", "198_b8euyKNT2wY", "1026_kJ_8F7YIEg4",
                            "1265_PBn1W-aOFUA", "1035_248bbw7mpdw", "1107_YXyd44eY_VY",
                            "1154_B4zEa_7Ejtk", "1282_cSmDnZFwqIM", "1477_GkuOGCQiUlk",
                            "2051_e0EI-QqHPIA"]

        with open(os.path.join(self.root, 'label_num_dic_final.json'), 'r') as f:
            self.categories_dict = json.load(f)
            self.categories_dict = {int(value): key for key, value in self.categories_dict.items()}

        self.class_mapping = self._load_class_mapping(args.class_mapping)
        self.img_list = self._create_list(class_list)

    def _create_list(self, class_list):
        if max(class_list) <= 5:
            train_split = 0
        elif max(class_list) <= 10:
            train_split = 1
        elif max(class_list) <= 15:
            train_split = 2
        elif max(class_list) <= 20:
            train_split = 3

        imgs_list = []
        with open('lists/vspw/%d.txt'%train_split) as f:
            for line in f:
                seq = line.strip().split('/')[-3]
                flow_exists = os.path.exists(os.path.join(self.root,
                                            line.strip().replace('mask', 'flow').replace('png', 'pkl.pkl')))
                if seq not in self.ignore_seqs and flow_exists:
                    imgs_list.append(line.strip())
        return imgs_list

    def _load_flow(self, mask_path):
        flow_path = os.path.join(self.root, mask_path).replace('mask', 'flow').replace('png', 'pkl.pkl')
        with open(flow_path, 'rb') as f:
            flow = pickle.load(f)
        flow = np.float32(flow)
        return flow, flow_path

    def _load_image(self, mask_path):
        image_path = os.path.join(self.root, mask_path).replace('mask', 'origin').replace('png', 'jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)

        # CHeck if sequence changed
        seq_name = mask_path.split('/')[-3]
        pick_new_support = False
        if self.current_sequence != seq_name:
            pick_new_support = True

        self.current_sequence = seq_name
        return image, image_path, pick_new_support

    def _convert_labels(self, mask):
        converted_mask = np.zeros_like(mask)
        label_class = []

        class_names = {cat_id: self.categories_dict[cat_id] for cat_id in np.unique(mask) \
                            if cat_id in self.categories_dict}
        class_names = {cat_id: self.class_mapping[cat_name] if cat_name in self.class_mapping else cat_name\
                            for cat_id, cat_name in class_names.items()}
        class_names = {cat_id: cat_name for cat_id, cat_name in class_names.items() \
                        if cat_name in self.current_external_classes}

        for cls, cls_name in class_names.items():
            if cls_name not in self.external_classes:
                continue

            new_class = self.external_classes.index(cls_name) + 1
            converted_mask[mask==cls] = new_class
            label_class.append(new_class)
        return converted_mask, label_class

    def extract_seq_names(self, paths):
        seqs = [path.split('/')[-3] for path in paths]
        return seqs

    def _get_label(self, mask_path, image_shape):
        mask = cv2.imread(os.path.join(self.root, mask_path), 0)
        label, label_class = self._convert_labels(mask)
        return label_class, label
