import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from visdom_logger import VisdomLogger
from collections import defaultdict
from .dataset.dataset import get_val_loader
from .util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, main_process, \
                  batch_vid_consistencyGPU, compute_map

from .util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from .classifier import Classifier
from .model.model import get_model
import torch.distributed as dist
from tqdm import tqdm
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from .visu import make_episode_visualization, make_episode_visualization_cv2, \
                  make_keyframes_vis
from typing import Tuple
from src.davis_metrics import db_eval_boundary

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

def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:

    print(f"==> Running DDP checkpoint example on rank {rank}.")
    setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed + rank)
        np.random.seed(args.manual_seed + rank)
        torch.manual_seed(args.manual_seed + rank)
        torch.cuda.manual_seed_all(args.manual_seed + rank)
        random.seed(args.manual_seed + rank)

    # ========== Model  ==========
    model = get_model(args).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    root = get_model_dir(args)

    if args.ckpt_used is not None:
        filepath = os.path.join(root, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Data  ==========
    val_loader, _ = get_val_loader(args, split_type='test')

    # ========== Test  ==========
    if args.episodic_val or args.temporal_episodic_val:
        val_Iou, val_loss = episodic_validate(args=args,
                                              val_loader=val_loader,
                                              model=model,
                                              use_callback=(args.visdom_port != -1),
                                              suffix=f'test')
    if args.distributed:
        dist.all_reduce(val_Iou), dist.all_reduce(val_loss)
        val_Iou /= world_size
        val_loss /= world_size

    #cleanup()


def episodic_validate(args: argparse.Namespace,
                      val_loader: torch.utils.data.DataLoader,
                      model: DDP,
                      use_callback: bool,
                      suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Start testing')

    model.eval()
    # ========== Metrics initialization  ==========
    if type(args.image_size) == tuple or type(args.image_size) == list:
        H, W = args.image_size
    else:
        H, W = args.image_size, args.image_size
    model.module.set_feature_res((H, W))
    c = model.module.bottleneck_dim
    h = model.module.feature_res[0]
    w = model.module.feature_res[1]

    if not hasattr(args, 'vc_wins'):
        args.vc_wins = [15]

    if not hasattr(args, 'multires_classifier'):
        args.multires_classifier = False

    all_weights = {'tti': [1.0,'auto','auto','auto'], 'repri': [1.0,'auto','auto',0.0]}
    if hasattr(args,'selected_weights') and len(args.selected_weights) != 0:
        all_weights = {'quickval': args.selected_weights}
    runtimes = {k: torch.zeros(args.n_runs) for k in all_weights.keys()}
    val_IoUs = {k: np.zeros(args.n_runs) for k in all_weights.keys()}
    val_Fscores = {k: np.zeros(args.n_runs) for k in all_weights.keys()}
    val_VCs = {}
    for method in all_weights.keys():
        val_VCs[method] = {kwin: np.zeros(args.n_runs) for kwin in args.vc_wins}
    val_losses = {k: np.zeros(args.n_runs) for k in all_weights.keys()}

    # ========== Perform the runs  ==========
    for run in tqdm(range(args.n_runs)):

        # =============== Initialize the metric dictionaries ===============
        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection = {k: defaultdict(int) for k in all_weights.keys()} # Default value is 0
        cls_union = {k: defaultdict(int)  for k in all_weights.keys()}
        cls_vc = {k: defaultdict(int)  for k in all_weights.keys()}
        cls_n_vc = {k: defaultdict(int)  for k in all_weights.keys()}

        IoU = {k: defaultdict(int) for k in all_weights.keys()}
        Fscores = {k: defaultdict(int) for k in all_weights.keys()}
        cls_n_fsc = {k: defaultdict(int)  for k in all_weights.keys()}

        # =============== episode = group of tasks ===============
        runtime = {k: 0  for k in all_weights.keys()}
        for qry_img, q_label, spprt_imgs, s_label, subcls, misc, paths  in tqdm(val_loader):
            t0 = time.time()

            # ====================== Only used for quick validation during training =============
            if 'quickval' in all_weights:
                skip = 5
                qry_img = qry_img[:, ::skip]
                q_label = q_label[:, ::skip]

                max_frames = 20
                qry_img = qry_img[:, :max_frames]
                q_label = q_label[:, :max_frames]

            # =========== Generate tasks and extract features for each task ===============
            with torch.no_grad():
                all_sprt = {'imgs': [], 'masks': [], 'paths': []}
                all_qry = {'imgs': [], 'masks': [], 'flows': [], 'paths': None}

                q_label = q_label.to(dist.get_rank(), non_blocking=True)
                spprt_imgs = spprt_imgs.to(dist.get_rank(), non_blocking=True)
                s_label = s_label.to(dist.get_rank(), non_blocking=True)
                qry_img = qry_img.to(dist.get_rank(), non_blocking=True)

                f_s = model.module.extract_features(spprt_imgs.squeeze(0))

                if qry_img.ndim > 4:
                    # B x N x C x H x W --> N x C X H X W
                    qry_img = qry_img.squeeze(0) # Squeeze batch dim
                    q_label = q_label.squeeze(0) # Squeeze batch dim

                #f_q = model.module.extract_features(qry_img)
                clip_len = args.inference_clip_len
                nclips = int(np.ceil(qry_img.shape[0]/clip_len))
                f_q = []
                for clip_idx in range(nclips):
                    if clip_idx == nclips - 1:
                        f_q_clip = model.module.extract_features(qry_img[clip_idx*clip_len:(clip_idx+1)*clip_len])
                    else:
                        f_q_clip = model.module.extract_features(qry_img[clip_idx*clip_len:(clip_idx+1)*clip_len])
                    f_q.append(f_q_clip.detach())
                    torch.cuda.empty_cache()
                f_q = torch.cat(f_q, dim=0)

                Nframes = f_q[0].size(0)
                shot = f_s[0].size(0)
                iter_num += Nframes

                n_shots = torch.tensor([shot] * Nframes).to(dist.get_rank(), non_blocking=True)
                n_layers = len(f_q)
                features_s = []
                features_q = []
                for layerno in range(n_layers):
                    features_s.append(f_s[layerno].repeat(Nframes, 1, 1, 1, 1).detach())
                    features_q.append(f_q[layerno].detach().unsqueeze(1))

                gt_s = s_label.repeat(Nframes, 1, 1, 1)
                gt_q = q_label.unsqueeze(1)
                classes = [class_.item() for class_ in subcls] * Nframes
                seqs = np.array(misc * Nframes)

                if args.visu:
                    all_sprt['imgs'] = spprt_imgs.cpu().numpy()
                    all_sprt['masks'] = s_label.cpu().numpy()
                    all_sprt['paths'] = [p[0] for p in paths]

                    all_qry['imgs'] = qry_img.cpu().numpy()
                    all_qry['masks'] = q_label.cpu().numpy()
                    all_qry['paths'] = os.path.join(val_loader.dataset.img_dir, misc[0])
                    all_qry['paths'] = [os.path.join(all_qry['paths'], fname) for fname in sorted(os.listdir(all_qry['paths'])) ]


            # =========== Normalize features along channel dimension ===============
            if args.norm_feat:
                for layerno in range(n_layers):
                    features_s[layerno] = F.normalize(features_s[layerno], dim=2)
                    features_q[layerno] = F.normalize(features_q[layerno], dim=2)

            for method, weights in all_weights.items():
                # =========== Create a callback is args.visdom_port != -1 ===============
                callback = VisdomLogger(port=args.visdom_port, env=args.visdom_env) if use_callback else None

                # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
                if args.multires_classifier:
                    classifier = [Classifier(args, model.module.classifier_chs[i]) for i in range(2)]
                else:
                    classifier = Classifier(args)

                probas_multires = []
                for layerno in range(len(classifier)):
                    classifier[layerno].init_prototypes(features_s[layerno], features_q[layerno], gt_s, gt_q, classes, callback, seqs=seqs)
                    batch_deltas = classifier[layerno].compute_FB_param(features_q=features_q[layerno], gt_q=gt_q, seqs=seqs)

                    # =========== Perform TTI inference ===============
                    batch_deltas = classifier[layerno].TTI(features_s[layerno], features_q[layerno], gt_s, gt_q,
                                                           classes, n_shots, seqs, callback,
                                                           weights=weights, adap_kshot=args.adap_kshot)
                    t1 = time.time()
                    runtime[method] += t1 - t0
                    logits = classifier[layerno].get_logits(features_q[layerno], seqs=seqs)  # [n_tasks, shot, h, w]
                    logits = F.interpolate(logits,
                                           size=(H, W),
                                           mode='bilinear',
                                           align_corners=True)

                    probas = classifier[layerno].get_probas(logits).detach()
                    if args.visu_keyframes:
                        root = os.path.join('plots', 'episodes', method, 'split_%d'%args.train_split)
                        os.makedirs(root, exist_ok=True)
                        save_path = os.path.join(root, f'run_{run}_iter_{iter_num}.png')
                        make_keyframes_vis(probas=probas, img_q=all_qry['imgs'].copy(),
                                           f_q=features_q[layerno], f_s=features_s[layerno], gt_s=gt_s, gt_q=gt_q,
                                           paths=all_qry['paths'], save_path=save_path)

                    if args.refine_keyframes_ftune and method == "tti":
                        # gt_q is only used to identify valid pixels and remove ones from padding for aug.
                        classifier[layerno].ftune_selected_keyframe(all_probas=probas, all_f_q=features_q[layerno],
                                                                    all_f_s=features_s[layerno], all_gt_s=gt_s,
                                                                    all_gt_q=gt_q, seqs=seqs, refine_oracle=args.refine_oracle)

                        logits = classifier[layerno].get_logits(features_q[layerno], seqs=seqs)  # [n_tasks, shot, h, w]
                        logits = F.interpolate(logits,
                                               size=(H, W),
                                               mode='bilinear',
                                               align_corners=True)
                        probas = classifier[layerno].get_probas(logits).detach()

                    probas_multires.append(probas)

                hr_res = probas_multires[-1].shape[-2:]
                probas = [F.interpolate(proba, hr_res) for proba in probas_multires]
                probas = torch.stack(probas).mean(dim=0)

                #np.save(f'{method}_probas.npy', probas.detach().cpu())
                intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2)  # [n_tasks, shot, num_class]
                intersection, union = intersection.cpu(), union.cpu()

                if args.tloss_type == 'temporal_repri' and args.enable_log:
                    log_temporal_repri(intersection, union, gt_q, features_q, probas, misc, method)

                if args.eval_vc:
                    video_consistency = batch_vid_consistencyGPU(seqs, probas, gt_q, 2,
                                                                 args.vc_size_th, args.vc_wins)
                    video_consistency = {k: v.cpu() for k, v in video_consistency.items()}
                else:
                    video_consistency = {'3': torch.zeros(len(seqs), 1, 2)}

                # ================== Log metrics ==================
                one_hot_gt = to_one_hot(gt_q, 2)
                valid_pixels = gt_q != 255
                loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
                loss_meter.update(loss.item())

                visited_seqs = []
                for i, class_ in enumerate(classes):
                    cls_intersection[method][class_] += intersection[i, 0, 1]  # Do not count background
                    cls_union[method][class_] += union[i, 0, 1]

                    proba = probas[i].argmax(1).squeeze(0).detach().cpu().numpy()
                    gt = gt_q[i].squeeze(0).cpu().numpy()
                    Fscores[method][class_] += db_eval_boundary(proba, gt)
                    cls_n_fsc[method][class_] += 1

                    if seqs[i] not in visited_seqs:
                        visited_seqs.append(seqs[i])
                        for kwin in video_consistency.keys():
                            if kwin not in cls_vc[method]:
                                cls_vc[method][kwin] = defaultdict(int)
                            cls_vc[method][kwin][class_] += video_consistency[kwin][i, 0, 1]
                        cls_n_vc[method][class_] += 1

                for class_ in cls_union[method]:
                    IoU[method][class_] = cls_intersection[method][class_] / (cls_union[method][class_] + 1e-10)
                    #Fscores[method][class_] = 2 * cls_intersection[method][class_] / \
                    #                            (cls_union[method][class_] + cls_intersection[method][class_] + 1e-10)

                if (iter_num % 200 == 0):
                    mIoU = np.mean([IoU[method][i] for i in IoU[method]])
                    print('Test: [{}/{}] '
                          'mIoU {:.4f} '
                          'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
                                                                                     args.test_num,
                                                                                     mIoU,
                                                                                     loss_meter=loss_meter,
                                                                                     ))

                # ================== Visualization ==================
                if args.visu:
                    for i in range(Nframes):
                        root = os.path.join(args.vis_dir, 'episodes', method, 'split_%d'%args.train_split)
                        os.makedirs(root, exist_ok=True)
                        save_path = os.path.join(root, f'run_{run}_iter_{iter_num}_{i:05d}.png')
                        flow_q = None

                        make_episode_visualization_cv2(img_s=all_sprt['imgs'][0].copy(),
                                                   img_q=all_qry['imgs'][i].copy(),
                                                   gt_s=all_sprt['masks'][0].copy(),
                                                   gt_q=all_qry['masks'][i].copy(),
                                                   path_s=all_sprt['paths'],
                                                   path_q=all_qry['paths'][i],
                                                   preds=probas[i].cpu().numpy().copy(),
                                                   save_path=save_path,
                                                   flow_q=flow_q)

        # ================== Evaluation Metrics on ALl episodes ==================
        for method in all_weights.keys():
            print('========= Method {}==========='.format(method))
            runtimes[method][run] = runtime[method] / float(len(val_loader))
            mIoU = np.mean(list(IoU[method].values()))

            for class_ in Fscores[method]:
                Fscores[method][class_] /= cls_n_fsc[method][class_]

            fscore = np.mean(list(Fscores[method].values()))

            for kwin in cls_vc[method].keys():
                for class_ in cls_vc[method][kwin].keys():
                   cls_vc[method][kwin][class_] /= (cls_n_vc[method][class_] + 1e-10)

            vc = {kwin: np.mean(list(cls_vc[method][kwin].values())) for kwin in cls_vc[method].keys()}
            print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
            for class_ in cls_union[method]:
                print("Class {} : {:.4f}".format(class_, IoU[method][class_]))

            val_Fscores[method][run] = fscore
            val_IoUs[method][run] = mIoU
            for kwin in vc.keys():
                val_VCs[method][kwin][run] = vc[kwin]
            val_losses[method][run] = loss_meter.avg

    # ================== Save metrics ==================
    for method in all_weights.keys():
        str_weights = str(all_weights[method])
        print(f'========Final Evaluation of {method} with weights {str_weights}============')
        print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs[method].mean()))
        print('Average Fscore over {} runs --- {:.4f}.'.format(args.n_runs, val_Fscores[method].mean()))

        vc_text = 'Average Vconsistency over {} runs ---'.format(args.n_runs)
        for kwin in val_VCs[method].keys():
            kwin_vc_mean = np.mean(val_VCs[method][kwin])
            vc_text += ' (' + str(kwin) + ',' + str(kwin_vc_mean)+ ') '
        print(vc_text)
        print('Average runtime / seq --- {:.4f}.'.format(runtimes[method].mean()))

    # This method works on multiple weights can not be used outside
    if 'quickval' in all_weights:
        return torch.tensor(np.mean(list(IoU['quickval'].values()))), torch.tensor(np.mean(val_losses['quickval']))
    return None, None


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    # TODO: Cleanup unnecessary distributed in inference not used
    args.distributed = distributed
    args.port = find_free_port()
    main_worker(0, world_size, args)
