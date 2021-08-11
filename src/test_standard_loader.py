import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .util import load_cfg_from_cfg_file, merge_cfg_from_list
from .dataset.dataset import get_val_loader, get_train_loader
from src.flow_viz import flow_to_image

def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--train_flag', action="store_true")
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

if __name__ == "__main__":
    args, train_flag = parse_args()

    # ========== Data  ==========
    train_loader, _ = get_train_loader(args)
    val_loader, _ = get_val_loader(args)
    if train_flag:
        loader = train_loader
    else:
        loader = val_loader

    train_classes = train_loader.dataset.class_list_names
    val_classes = val_loader.dataset.class_list_names

    #intersect = [v for k, v in train_classes.items() if v in val_classes.values()]
    #assert len(intersect) == 0, "Overlap in classes"

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    print(loader.dataset.class_list_names)
    colors = loader.dataset.class_colors
    print(colors)
    print(len(loader))
    for i, (images, labels)  in enumerate(tqdm(loader)):
#        pass
        for bsz in range(images.shape[0]):
            for frame in range(images.shape[1]):
                img = denormalize(images[bsz, frame], args.mean, args.std)
                cv2.imwrite("tmp/img_%05d.png"%i, img)
                #cv2.imshow("image", img)

                lbl = labels[bsz, frame, 0].numpy()
                colored_lbl = np.zeros((*lbl.shape, 3), dtype=np.uint8)
                for cls in np.unique(lbl):
                    if cls == 0:
                        continue
                    colored_lbl[lbl==cls] = colors[cls]
                cv2.imwrite("tmp/lbl_%05d.png"%i, colored_lbl)
                #cv2.imshow("label", colored_lbl)

                ch = cv2.waitKey(10)
                if ch %256 == ord('q'):
                    break
