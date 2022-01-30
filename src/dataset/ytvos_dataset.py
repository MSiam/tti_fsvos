import torch
import cv2
from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image
import src.dataset.ytvos_transform as transform
import argparse

from src.util import get_split_base_protos
from src.dataset.aux_dataset import YTVOSAuxiliaryData

class YTVOSBase(Dataset):
    def __init__(self,
                 transform: transform.Compose,
                 train: bool,
                 args: argparse.Namespace):

        self.train = train
        self.every_frame = args.every_frame
        self.n_frames = args.n_frames

        if self.train or args.test_split == 'default':
            self.set_index = args.train_split
        else:
            self.set_index = args.test_split

        self.transform = transform
        self.img_dir = os.path.join(args.data_root, 'train', 'JPEGImages')
        self.ann_file = os.path.join(args.data_root, 'train', 'train.json')

        self.load_annotations()

        print('data set index: ', self.set_index)
        self.train_list = [n + 1 for n in range(40) if n % 4 != (self.set_index)]
        self.valid_list = [n + 1 for n in range(40) if n % 4 == (self.set_index)]

        if self.train:
            self.class_list = self.train_list
        else:
            self.class_list = self.valid_list

    def load_class_list_vids(self):
        self.video_ids = []
        for class_id in self.class_list:
            tmp_list = self.ytvos.getVidIds(catIds=class_id)
            tmp_list.sort()
            self.video_ids.append(tmp_list)  # list[list[video_id]]
        if not self.train:
            self.test_video_classes = []
            for i in range(len(self.class_list)):
                for j in range(len(self.video_ids[i]) - self.support_frame):  # remove the support set
                    self.test_video_classes.append(i)

    def load_annotations(self):
        self.ytvos = YTVOS(self.ann_file)
        self.vid_ids = self.ytvos.getVidIds()  # list[2238] begin : 1
        self.vid_infos = self.ytvos.vids  # vids
        for vid, vid_info in self.vid_infos.items():  # for each vid
            vid_name = vid_info['file_names'][0].split('/')[0]  # '0043f083b5'
            vid_info['dir'] = vid_name
            frame_len = vid_info['length']  # int
            frame_object, frame_class = [], []
            for i in range(frame_len): frame_object.append([])
            for i in range(frame_len): frame_class.append([])
            category_set = set()
            annos = self.ytvos.vidToAnns[vid]  # list[]
            for anno in annos:  # instance_level anns
                assert len(anno['segmentations']) == frame_len, (
                vid_name, len(anno['segmentations']), vid_info['length'])
                for frame_idx in range(frame_len):
                    anno_segmentation = anno['segmentations'][frame_idx]
                    if anno_segmentation is not None:
                        frame_object[frame_idx].append(anno['id'])  # add instance to vid_frame
                        frame_class[frame_idx].append(anno['category_id'])  # add instance class to vid_frame
                        category_set = category_set.union({anno['category_id']})
            vid_info['objects'] = frame_object
            vid_info['classes'] = frame_class
            class_frame_id = dict()
            for class_id in category_set:  # frames index for each class
                class_frame_id[class_id] = [i for i in range(frame_len) if class_id in frame_class[i]]
            vid_info['class_frames'] = class_frame_id

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def get_class_list(self):
        return self.class_list

class YTVOSStandard(YTVOSBase):
    def __init__(self, **kwargs):
        super(YTVOSStandard, self).__init__(**kwargs)
        # In standard training training and val classes are the same
        self.nrep = 10
        self.class_list = self.train_list
        self.support_frame = 0
        self.load_class_list_vids()

        train_video_ids = []
        val_video_ids = []
        for vids in self.video_ids:
            n = int(len(vids) * 0.9)
            train_video_ids += [v for v in vids[:n] if v not in val_video_ids]
            val_video_ids += [v for v in vids[n:] if v not in train_video_ids]

        self.use_aux = False
        args = kwargs['args']
        if hasattr(args, 'aux_train_name') and self.train:
            self.use_aux = True
            # For the Auxiliary dataset only images are used and thus can use the val set
            self.aux_train_data = YTVOSAuxiliaryData(transform=kwargs['transform'], args=kwargs['args'],
                                                    class_list=self.valid_list)

        assert len(np.intersect1d(val_video_ids, train_video_ids)) == 0, \
                    "Train and Val seqs no intersection"

        if self.train:
            self.combined_video_ids = train_video_ids
        else:
            self.combined_video_ids = val_video_ids

        self.class_list_names = {i+1: self.ytvos.cats[cls]['name'] \
                                    for i, cls in enumerate(self.class_list)}
        self.class_colors = {i+1: np.random.random((1,3))*255 for i in range(len(self.class_list))}

    def __getitem__(self, idx):
        vid = self.combined_video_ids[idx%len(self.combined_video_ids)]
        frames, masks = self.get_frames_labels(vid)

        if self.transform is not None:
            frames, masks = self.transform(frames, masks)
        masks = masks[:, 0].long()

        if self.use_aux:
            aux_frames, aux_labels, _ = self.aux_train_data.__getitem__(idx)
            frames = {'aux_images': aux_frames, 'images': frames}
            masks = {'aux_labels': aux_labels, 'labels': masks}

        return frames, masks, self.use_aux

    def get_frames_labels(self, vid):
        vid_info = self.vid_infos[vid]
        frame_lists = vid_info['class_frames']
        frame_list = []
        for k, v in frame_lists.items():
            if k not in self.class_list:
                continue
            frame_list += v

        frame_list = list(set(frame_list))

        frame_len = len(frame_list)
        orig_choice_frame = [frame_list[n] for n in range(0, len(frame_list), self.every_frame)]
        if len(orig_choice_frame) > self.n_frames:
            choice_frame = random.choices(orig_choice_frame, k=self.n_frames)
        elif len(orig_choice_frame) < self.n_frames:
            choice_frame = orig_choice_frame
            while len(choice_frame) < self.n_frames:
                choice_frame = choice_frame + [random.choice(orig_choice_frame)]
        else:
            choice_frame = orig_choice_frame

        masks = []
        frames = []
        for frame_id in choice_frame:
            object_ids = vid_info['objects'][frame_id]
            mask = None
            for object_id in object_ids:
                ann = self.ytvos.loadAnns(object_id)[0]
                if ann['category_id'] not in self.class_list:
                    continue
                cls_map = self.class_list.index(ann['category_id']) + 1
                temp_mask = self.ytvos.annToMask(ann, frame_id)
                if mask is None:
                    mask = temp_mask
                mask[temp_mask==1] = cls_map

            assert mask is not None, "Wrong frame included"
            masks.append(mask)
            frames.append(np.array(Image.open(os.path.join(self.img_dir,
                  vid_info['file_names'][frame_id]))) )
        assert len(frames) == self.n_frames, "# Frames is not correct"
        return frames, masks

    def __len__(self):
        return len(self.combined_video_ids) * self.nrep

class YTVOSEpisodic(YTVOSBase):
    def __init__(self,
                 transform: transform.Compose,
                 train: bool,
                 args: argparse.Namespace):
        super(YTVOSEpisodic, self).__init__(transform=transform,
                                            train=train,
                                            args=args)
        self.support_frame = args.shot
        self.query_frame = args.query_frame
        self.sample_per_class = args.sample_per_class

        self.multi_rnd_sprt = False
        if hasattr(args, 'multi_rnd_sprt'):
            self.multi_rnd_sprt = args.multi_rnd_sprt

        self.load_class_list_vids()

        if self.train:
            self.length = len(self.class_list) * self.sample_per_class
        else:
            self.length = len(self.test_video_classes)  # test

        self.sequence_support = None
        self.class_list_names = {i+1: self.ytvos.cats[cls]['name'] \
                                    for i, cls in enumerate(self.class_list)}
        self.class_colors = {i+1: np.random.random((1,3))*255 for i in range(len(self.class_list))}

    def get_GT_byclass(self, vid, class_id, frame_num=1, test=False):
        vid_info = self.vid_infos[vid]
        frame_list = vid_info['class_frames'][class_id]
        frame_len = len(frame_list)
        choice_frame = random.sample(frame_list, 1)
        if test:
            frame_num = frame_len
        if frame_num > 1:
            if frame_num <= frame_len:
                choice_idx = frame_list.index(choice_frame[0])
                if choice_idx < frame_num:
                    begin_idx = 0
                    end_idx = frame_num
                else:
                    begin_idx = choice_idx - frame_num + 1
                    end_idx = choice_idx + 1
                choice_frame = [frame_list[n] for n in range(begin_idx, end_idx)]
            else:
                choice_frame = []
                for i in range(frame_num):
                    if i < frame_len:
                        choice_frame.append(frame_list[i])
                    else:
                        choice_frame.append(frame_list[frame_len - 1])
        frames = [np.array(Image.open(os.path.join(self.img_dir, vid_info['file_names'][frame_idx]))) for frame_idx in
                  choice_frame]
        paths = [os.path.join(self.img_dir, vid_info['file_names'][frame_idx]) for frame_idx in choice_frame]

        masks = []
        for frame_id in choice_frame:
            object_ids = vid_info['objects'][frame_id]
            mask = None
            for object_id in object_ids:
                ann = self.ytvos.loadAnns(object_id)[0]
                if ann['category_id'] not in self.class_list:
                    continue
                track_id = 1
                if ann['category_id'] != class_id:
                    track_id = 0
                temp_mask = self.ytvos.annToMask(ann, frame_id)
                if mask is None:
                    mask = temp_mask * track_id
                else:
                    mask += temp_mask * track_id

            assert mask is not None
            mask[mask > 0] = 1
            masks.append(mask)

        return frames, masks, paths

    def __gettrainitem__(self, idx):
        list_id = idx // self.sample_per_class
        vid_set = self.video_ids[list_id]

        query_vid = random.sample(vid_set, 1)
        support_vid = random.sample(vid_set, self.support_frame)

        query_frames, query_masks, _ = self.get_GT_byclass(query_vid[0], self.class_list[list_id], self.query_frame)

        support_frames, support_masks, sprt_paths = [], [], []
        for i in range(self.support_frame):
            one_frame, one_mask, one_path = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
            support_frames += one_frame
            support_masks += one_mask
            sprt_paths += one_path

        if self.transform is not None:
            query_frames, query_masks = self.transform(query_frames, query_masks)
            support_frames, support_masks = self.transform(support_frames, support_masks)

        query_masks = query_masks.squeeze(1).long()
        support_masks = support_masks.squeeze(1).long()
        return query_frames, query_masks, support_frames, support_masks, self.class_list[list_id], [], []

    def __gettestitem__(self, idx):
        # random.seed()
        if self.multi_rnd_sprt:
            # Ensures random support set with each sequence
            begin_new = True
        else:
            # What DANet performs
            # Ensures random support set with all seqs per class
            begin_new = False
            if idx == 0:
                begin_new = True
            else:
                if self.test_video_classes[idx] != self.test_video_classes[idx - 1]:
                    begin_new = True

        list_id = self.test_video_classes[idx]
        vid_set = self.video_ids[list_id]

        support_frames, support_masks, sprt_paths = [], [], []
        if begin_new:
            support_vid = random.sample(vid_set, self.support_frame)
            query_vids = []
            for id in vid_set:
                if not id in support_vid:
                    query_vids.append(id)
            self.query_ids = query_vids
            self.query_idx = -1
            for i in range(self.support_frame):
                one_frame, one_mask, one_path = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
                support_frames += one_frame
                support_masks += one_mask
                sprt_paths += one_path
        else:
            support_frames, support_masks, sprt_paths = self.sequence_support

        if self.multi_rnd_sprt:
            query_vid = random.choice(self.query_ids)
        else:
            self.query_idx += 1
            query_vid = self.query_ids[self.query_idx]

        query_frames, query_masks, _ = self.get_GT_byclass(query_vid, self.class_list[list_id], test=True)
        if self.transform is not None:
            query_frames, query_masks = self.transform(query_frames, query_masks)
            # TODO: remove AddAxis from transforms
            query_masks = query_masks.squeeze(1).long() # squeeze channel 1 -> N x H x W

            if begin_new:
                support_frames, support_masks = self.transform(support_frames, support_masks)
                support_masks = support_masks.squeeze(1).long() # squeeze channel 1 -> K x H x W
                self.sequence_support = (support_frames, support_masks, sprt_paths)

        vid_info = self.vid_infos[query_vid]
        vid_name = vid_info['dir']
        return query_frames, query_masks, support_frames, support_masks, self.class_list[list_id], vid_name, sprt_paths

    def __getitem__(self, idx):
        if self.train:
            return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __len__(self):
        return self.length
