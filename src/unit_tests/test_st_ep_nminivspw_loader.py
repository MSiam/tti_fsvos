import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.util import load_cfg_from_cfg_file, merge_cfg_from_list
from src.dataset.dataset import get_val_loader, get_train_loader

def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--train_flag', type=int, default=0) #0: train, 1:val, 2:test
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg, args.train_flag

def denormalize(img, mean, scale):
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,::-1]*255, np.uint8)
    return img

def map_label(lbl, colors):
    colored_lbl = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    for cls in np.unique(lbl):
        if cls == 0:
            continue
        elif cls == 255:
            colored_lbl[lbl==cls] = np.array([255, 255, 0])
        else:
            colored_lbl[lbl==cls] = colors[cls]
    return colored_lbl

if __name__ == "__main__":
    args, train_flag = parse_args()
    args.distributed = False

    # ========== Data  ==========
    train_loader, _ = get_train_loader(args)
    val_loader, _ = get_val_loader(args, 'val')
    test_loader, _ = get_val_loader(args, 'test')

    if train_flag == 0:
        loader = train_loader
        colors = loader.dataset.class_colors
    elif train_flag == 1:
        loader = val_loader
        colors = {1: np.array([255, 255, 255])}
    else:
        loader = test_loader
        colors = {1: np.array([255, 255, 255])}

    train_classes = train_loader.dataset.class_list_names
    val_classes = val_loader.dataset.class_list_names
    test_classes = test_loader.dataset.class_list_names

    intersect = [cls for cls in train_classes if cls in val_classes]
    assert len(intersect) == 0, "Overlap in classes train, val"

    intersect = [cls for cls in train_classes if cls in test_classes]
    assert len(intersect) == 0, "Overlap in classes train, test"

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    print(loader.dataset.class_list_names)
    print(colors)
    print(len(loader))
    if train_flag == 0:
        for i, (images, labels)  in enumerate(tqdm(loader)):
            for bsz in range(images.shape[0]):
                img = denormalize(images[bsz], args.mean, args.std)
                cv2.imwrite("tmp/img_%05d.png"%i, img)

                lbl = labels[bsz, 0].numpy()
                colored_lbl = map_label(lbl, colors)
                cv2.imwrite("tmp/lbl_%05d.png"%i, colored_lbl)

    else: # Peisodic Val/Test
        for i, (qry_img, q_label, spprt_imgs, s_label, chosen_cls)  in enumerate(tqdm(loader)):
            qry_img = qry_img[0] # Batch is always 1
            q_label = q_label[0]

            for shot in range(len(spprt_imgs)):
                for frame in range(spprt_imgs[shot].shape[1]):
                    sprt_img = denormalize(spprt_imgs[shot][0, frame], args.mean, args.std)
                    cv2.imwrite("tmp/sprt_img_%05d_%02d_%03d.png"%(i, shot, frame), sprt_img)

                    sprt_lbl = s_label[shot][0, frame, 0]
                    sprt_colored_lbl = map_label(sprt_lbl, colors)
                    cv2.imwrite("tmp/sprt_lbl_%05d_%02d_%03d.png"%(i, shot, frame), sprt_colored_lbl)

            for frame in range(qry_img.shape[1]):
                img = denormalize(qry_img[0, frame], args.mean, args.std)
                cv2.imwrite("tmp/img_%05d_%03d.png"%(i, frame), img)
                #cv2.imshow("image", img)

                lbl = q_label[0, frame, 0].numpy()
                colored_lbl = map_label(lbl, colors)
                cv2.imwrite("tmp/lbl_%05d_%03d.png"%(i, frame), colored_lbl)
