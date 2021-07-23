import argparse
import os
import torch
import cv2
import numpy as np

from .util import load_cfg_from_cfg_file, merge_cfg_from_list
from .dataset.dataset import get_val_loader
from src.flow_viz import flow_to_image

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

    # ========== Data  ==========
    episodic_val_loader, _ = get_val_loader(args)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    # 958, 1016, 11533, 153
    for i, (qry_img, q_label, spprt_imgs, s_label, subcls, _, image_path)  in enumerate(episodic_val_loader):
        print(image_path)
        if args.flow_aggregation:
            qry_flow = qry_img['flow']
            qry_img = qry_img['image']
            flow_img = flow_to_image(qry_flow[0].numpy().transpose(1,2,0))
            cv2.imwrite("tmp/qry_flow_%05d.png"%i, flow_img)

        qry_img = denormalize(qry_img[0], args.mean, args.std)
        cv2.imwrite("tmp/qry_img_%05d.png"%i, qry_img)


        q_label = q_label[0]
        q_label[q_label==255] = 0
        q_label = q_label * 255

        cv2.imwrite("tmp/qry_lbl_%05d.png"%i, q_label.numpy())

        for k, (sprt_img, sprt_lbl) in enumerate(zip(spprt_imgs, s_label)):
            sprt_img = denormalize(sprt_img[0], args.mean, args.std)
            cv2.imwrite("tmp/sprt_img_shot%d_%05d.png"%(k,i), sprt_img)

            sprt_lbl = sprt_lbl[0]
            sprt_lbl[sprt_lbl==255] = 0
            sprt_lbl = sprt_lbl * 255
            cv2.imwrite("tmp/sprt_lbl_shot%d_%05d.png"%(k,i), sprt_lbl.numpy())
