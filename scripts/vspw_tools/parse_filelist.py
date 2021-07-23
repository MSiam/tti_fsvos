from tqdm import tqdm
import json
import argparse
import os
import cv2
import numpy as np


def create_list(root, split):
    sequences = sorted(os.listdir(os.path.join(root, 'data')))

    indices = {0: range(5), 1: range(5, 10) , 2: range(10, 15), 3: range(15, 20)}

    with open(os.path.join(root, 'label_num_dic_final.json'), 'r') as f:
        categories_dict = json.load(f)
        categories_dict = {int(value): key for key, value in categories_dict.items()}

    with open('../../lists/pascal/vspw2pascal_class_mapping.json', "r") as f:
        class_mapping = json.load(f)

    with open('../../lists/pascal/classes.json', "r") as f:
        classes = np.array(json.load(f))
        current_external_classes = classes[indices[split]]

    with open(os.path.join(root, 'vspw_videolist.txt'), 'r') as f:
        considered_seqs = []
        for line in f:
            sq = line.split(' ')[0]
            if os.path.exists(os.path.join(root, 'data', sq, 'origin')):
                considered_seqs.append(line.split(' ')[0])

    imgs_list = []
    for seq in tqdm(sequences):
        if seq not in considered_seqs:
            continue

        masks_path = os.path.join(root, 'data', seq, 'mask')
        for frame in sorted(os.listdir(masks_path)):
            if frame[0] == '.':
                continue

            mask = cv2.imread(os.path.join(masks_path, frame), 0)
            frame_classes = [categories_dict[cat_id] for cat_id in np.unique(mask) \
                             if cat_id in categories_dict]
            frame_classes = [class_mapping[cat_name] if cat_name in class_mapping else cat_name\
                                for cat_name in frame_classes]

            if len(np.intersect1d(frame_classes, current_external_classes)) != 0:
                imgs_list.append(os.path.join(masks_path, frame).replace(root, ''))
    return imgs_list

def write_file(fname, filelist):
    with open(fname, 'w') as f:
        for file_ in filelist:
            f.write(file_+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parse vspw")
    parser.add_argument("--root", type=str, default="/local/riemann/home/msiam/VSPW/")
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()

    filelist = create_list(args.root, args.split)
    write_file('../../lists/vspw/%d.txt'%args.split, filelist)
