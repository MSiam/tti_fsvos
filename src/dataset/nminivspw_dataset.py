import argparse
from typing import List
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import torch

import src.dataset.transform as transform
from src.dataset.static_dataset import StandardData, EpisodicData

class NMiniVSPWStandardData(StandardData):
    def __init__(self, **kwargs):
        assert kwargs['train'], "New MiniVSPW Standard Loader only works for training"
        self.fold= kwargs['args'].train_split
        class_dic = self.load_classes(kwargs['args'].train_split, kwargs['data_list_path'])

        self.use_aux = False
        super(NMiniVSPWStandardData, self).__init__(**kwargs)

        self.class_list, self.class_list_names = list(class_dic.keys()), list(class_dic.values())
        if 'pascal' in kwargs['data_list_path']:
            correct_nclasses = 15
        else:
            correct_nclasses = 30
        assert len(self.class_list) == correct_nclasses, "# Training Classes should be" + str(correct_nclasses) + \
                                                         ", but are " + str(len(self.class_list))

        self.class_colors = {i+1: np.random.random((1,3))*255 for i in range(len(self.class_list))}

    def load_classes(self, foldno, list_fname):
        list_fname = list_fname.split('/')
        list_fname_ = ''
        for l in list_fname[:-1]:
            list_fname_ += l + '/'
        list_fname_ += ('class_' + list_fname[-1])
        clsfname = list_fname_.replace('.txt', '_%d.json'%self.fold)
        with open(clsfname, 'r') as f:
            classes = json.load(f)
        classes = {int(k): v for k, v in classes.items()}
        return classes

    def load_filenames(self, data_root, data_list_path, class_list):
        """
        Load filenames from MiniVSPW for the intended fold
        class_list: is not used in this loader as it is set in the constructor
        """
        data_list_path = data_list_path.replace('.txt', '_%d.txt'%self.fold)
        data_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                data_list.append(os.path.join(data_root, line.split(' ')[1].strip()) )
        return data_list, None

    def __getitem__(self, index):
        mask_fname = self.data_list[index]
        img_fname = mask_fname.replace('png', 'jpg').replace('mask', 'origin')

        img = np.array(Image.open(img_fname))
        mask = np.array(Image.open(mask_fname))

        lbl_class_mapping = {}
        for c in np.unique(mask):
            if c in self.class_list:
                lbl_class_mapping[c] = self.class_list.index(c) + 1
            else:
                lbl_class_mapping[c] = 0

        temp_mask = np.zeros_like(mask)
        for k, v in lbl_class_mapping.items():
            temp_mask[mask==k] = v
        mask = temp_mask
        assert len(np.unique(mask)) > 0, "Empty Label"

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask, self.use_aux


class NMiniVSPWEpisodicData(EpisodicData):
    def __init__(self, **kwargs):
        self.fold= kwargs['args'].train_split
        class_dic = self.load_classes(kwargs['args'].train_split, kwargs['data_list_path'])
        self.class_list_current = list(class_dic.keys())
        self.sprtset_as_frames = kwargs['args'].sprtset_as_frames
        super(NMiniVSPWEpisodicData, self).__init__(**kwargs)

        self.split_type = kwargs['args'].split_type
        self.class_list, self.class_list_names = list(class_dic.keys()), list(class_dic.values())
        if 'pascal' in kwargs['data_list_path']:
            correct_test_nclasses = 5
        else:
            correct_test_nclasses = 10
        assert len(self.class_list) == correct_test_nclasses, \
                "# Test/Val Classes in NMiniVSPW should be" + str(correct_test_nclasses)

        # Number of sequences without rep per fold in Test Split 0: 61, 1: 96, 2: 261 , 3: 64
        # Number of sequences without rep per fold in Val Split 0: 36, 1: 49, 2: 51, 3: 51
        nrep = 10 # Get on average 300-500 seq per fold
        if self.split_type == 'test' and self.fold in [0,1,3]:
            self.data_list = self.data_list * nrep
        #self.nsupport_frames = kwargs['args'].nsupport_frames

    def load_classes(self, foldno, list_fname):
        list_fname = list_fname.split('/')
        list_fname_ = ''
        for l in list_fname[:-1]:
            list_fname_ += l + '/'

        list_fname_ += ('class_' + list_fname[-1])
        clsfname = list_fname_.replace('.txt', '_%d.json'%self.fold)
        with open(clsfname, 'r') as f:
            classes = json.load(f)
        classes = {int(k): v for k, v in classes.items()}
        return classes

    def load_filenames(self, data_root, data_list_path, class_list):
        """
        Load filenames from MiniVSPW for the intended fold
        class_list: is not used in this loader as it is set in the constructor
        """
        data_list_path = data_list_path.replace('.txt', '_%d.txt'%self.fold)
        temp_data_list_path = ''
        tokens = data_list_path.split('/')
        for i, path_ in enumerate(tokens):
            if i == len(tokens) - 1:
                temp_data_list_path += '.' + path_.replace('txt', 'npy')
            else:
                temp_data_list_path += path_ + '/'

        if os.path.exists(temp_data_list_path):
            loaded_list = np.load(temp_data_list_path, allow_pickle=True).item()
            data_list = loaded_list['data_list']
            self.classes_per_seq = loaded_list['classes_per_seq']
            self.seqs_per_cls = loaded_list['seqs_per_cls']
        else:
            data_list = []
            self.classes_per_seq = {} # {seq: [cls1, cls2, ..], ..}
            self.seqs_per_cls = {cls: {} for cls in self.class_list_current} #{cls: [seq1, seq2, ..], ..}

            print("===> Processing Classes per Seq + Seqs per Cls")
            with open(data_list_path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    seq_name, fname = line.strip().split(' ')
                    mask = np.array(Image.open(os.path.join(data_root, fname)))

                    if seq_name not in self.classes_per_seq:
                        self.classes_per_seq[seq_name] = []

                    if seq_name not in data_list:
                        data_list.append(seq_name)

                    for cls in np.unique(mask):
                        if cls in self.class_list_current:
                            if cls not in self.classes_per_seq[seq_name]:
                                self.classes_per_seq[seq_name].append(cls)
                            if seq_name not in self.seqs_per_cls[cls]:
                                self.seqs_per_cls[cls][seq_name] = []
                            self.seqs_per_cls[cls][seq_name].append(fname)

            np.save(temp_data_list_path, {'data_list': data_list, 'seqs_per_cls': self.seqs_per_cls,
                                          'classes_per_seq': self.classes_per_seq})
        return data_list, None

    def _load_seq(self, seq_name, cls, files=None):
        seq_path = os.path.join(self.data_root, seq_name, 'origin')
        imgs = []
        masks = []

        if files is None:
            files = [os.path.join(seq_path, fname) for fname in sorted(os.listdir(seq_path))]
        else:
            files = [os.path.join(self.data_root, fname).replace('mask', 'origin').replace('png', 'jpg') for fname in files]

        for image_path in files:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.float32(image)
            imgs.append(image)

            mask = cv2.imread(
                    image_path.replace('origin', 'mask').replace('jpg', 'png'), 0
                )
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == cls] = 1
            masks.append(temp_mask)

        return imgs, masks

    def __getitem__(self, index):
        """
        returns:

            qry_frames: [1 x T x C x H x W]
            qry_masks: [1 x T x 1 x H x W]
            support_frames: [shot x T x C x H x W]
            support_masks: [shot x T x 1 x H x W]
            subcls_list: List[int] chosen classes
        """
        seq_name = self.data_list[index]
        classes = self.classes_per_seq[seq_name]

        chosen_class = np.random.choice(classes)
        qry_frames, qry_masks = self._load_seq(seq_name, chosen_class)
        if self.transform is not None:
            qry_frames, qry_masks = self.transform(qry_frames, qry_masks)
        qry_masks = qry_masks[:, 0].long()

        support_frames, support_masks = [], []
        selected_seqs = list(self.seqs_per_cls[chosen_class].keys())
        selected_seqs.remove(seq_name)

        for shot in range(self.shot):
            sprt_seq = np.random.choice(selected_seqs)
            sprt_frames, sprt_masks = self._load_seq(sprt_seq, chosen_class,
                                                     self.seqs_per_cls[chosen_class][sprt_seq])
            if self.transform is not None:
                sprt_frames, sprt_masks = self.transform(sprt_frames, sprt_masks)
            if not self.sprtset_as_frames:
                rnd_idx = np.random.randint(0, sprt_frames.shape[0])
                sprt_frames = sprt_frames[rnd_idx]
                sprt_masks = sprt_masks[rnd_idx]

            support_frames.append(sprt_frames)
            support_masks.append(sprt_masks.long())
        subcls_list = [chosen_class]

        if not self.sprtset_as_frames:
            support_frames = torch.stack(support_frames)
            support_masks = torch.stack(support_masks)
            support_masks = support_masks[:, 0]

        return qry_frames, qry_masks, support_frames, support_masks, subcls_list, seq_name, []
