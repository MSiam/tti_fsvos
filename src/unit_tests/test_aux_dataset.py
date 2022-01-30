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
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def denormalize(img, mean, scale):
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,::-1]*255, np.uint8)
    return img

if __name__ == "__main__":
    args = parse_args()
    args.distributed = False

    # ========== Data  ==========
    loader, _ = get_train_loader(args)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if not os.path.exists('tmp/aux_images'):
        os.mkdir('tmp/aux_images')
        os.mkdir('tmp/aux_labels')
        os.mkdir('tmp/images')
        os.mkdir('tmp/labels')

    colors = {i+1: np.random.random((1,3))*255 for i in range(len(loader.dataset.class_list))}

    print(len(loader))
    for i, (images, gt, aux_flag)  in enumerate(tqdm(loader)):
        aux_images = images['aux_images']
        images = images['images']

        aux_gt = gt['aux_labels']
        gt = gt['labels']

        for bsz in range(aux_images.shape[0]):
            if images.ndim == 4:
                images = images.unsqueeze(1)
                gt = gt.unsqueeze(1)

            for frame in range(images.shape[1]):

                img = denormalize(images[bsz, frame], args.mean, args.std)
                cv2.imwrite("tmp/images/%05d_%02d_%02d.png"%(i, bsz, frame), img)

                lbl = gt[bsz, frame].numpy()
                colored_lbl = np.zeros((*lbl.shape, 3), dtype=np.uint8)
                for cls in np.unique(lbl):
                    if cls == 0 or cls == 255:
                        continue
                    colored_lbl[lbl==cls] = colors[cls]
                cv2.imwrite("tmp/labels/%05d_%02d_%02d.png"%(i, bsz, frame), colored_lbl)

            for frame in range(aux_images.shape[1]):
                for window in range(aux_images.shape[2]):
                    img = denormalize(aux_images[bsz, frame, window], args.mean, args.std)
                    cv2.imwrite("tmp/aux_images/img_%05d_%02d_%02d_%02d.png"%(i, bsz, frame, window), img)

                    lbl = aux_gt[bsz, frame, window, 0].numpy()
                    colored_lbl = np.zeros((*lbl.shape, 3), dtype=np.uint8)
                    for cls in np.unique(lbl):
                        if cls == 0:
                            continue
                        if cls == 255:
                            colored_lbl[lbl==cls] = np.array((255, 0, 0))
                        else:
                            colored_lbl[lbl==cls] = colors[cls]
                    cv2.imwrite("tmp/aux_labels/lbl_%05d_%02d_%02d_%02d.png"%(i, bsz, frame, window), colored_lbl)


                    #cv2.imshow("image", img)
                    #ch = cv2.waitKey(10)
                    #if ch %256 == ord('q'):
                    #    break
