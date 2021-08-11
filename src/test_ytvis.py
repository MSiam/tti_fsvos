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
                  batch_vid_consistencyGPU

from .util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from .classifier import Classifier
from .model.pspnet import get_model
import torch.distributed as dist
from tqdm import tqdm
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from .visu import make_episode_visualization, make_episode_visualization_cv2
from typing import Tuple
from src.util import generate_roi_grid

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

def log_temporal_repri(intersection, union, gt_q, features_q, probas, seq_name, method):
    IoU = intersection[:, 0, 1] / union[:, 0, 1]
    valid_pixels_q = (gt_q != 255).float()
    marginal = (valid_pixels_q.unsqueeze(2) * probas).sum(dim=(1, 3, 4))
    marginal /= valid_pixels_q.sum(dim=(1, 2, 3)).unsqueeze(1)

    out_dir = os.path.join('dumped_marginals', method)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(os.path.join(out_dir, '%s.npy'%seq_name[0]), {'miou': IoU, 'marginal': marginal})

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
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ========== Data  ==========
    val_loader, _ = get_val_loader(args)

    # ========== Test  ==========
    if args.episodic_val or args.temporal_episodic_val:
        val_Iou, val_loss = episodic_validate(args=args,
                                              val_loader=val_loader,
                                              model=model,
                                              use_callback=(args.visdom_port != -1),
                                              suffix=f'test')
    else:
        mIoU, _ = standard_validate(args=args,
                                  val_loader=val_loader,
                                  model=model,
                                  use_callback=(args.visdom_port != -1),
                                  suffix=f'test')
        print("Mean IoU on Validation Set ", mIoU)

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

    all_weights = {'tti': [1.0,'auto','auto','auto'], 'repri': [1.0,'auto','auto',0.0]}

    runtimes = {k: torch.zeros(args.n_runs) for k in all_weights.keys()}
    val_IoUs = {k: np.zeros(args.n_runs) for k in all_weights.keys()}
    val_Fscores = {k: np.zeros(args.n_runs) for k in all_weights.keys()}
    val_VCs = {k: np.zeros(args.n_runs) for k in all_weights.keys()}
    val_losses = {k: np.zeros(args.n_runs) for k in all_weights.keys()}

    if args.tloss_type == 'fb_consistency':
        roi_grid = generate_roi_grid(h, w)
        consistency_type = args.consistency_type
    else:
        roi_grid = None
        consistency_type = None

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

        # =============== episode = group of tasks ===============
        runtime = {k: 0  for k in all_weights.keys()}
        for qry_img, q_label, spprt_imgs, s_label, subcls, misc, paths  in tqdm(val_loader):
            t0 = time.time()

            # =========== Generate tasks and extract features for each task ===============
            with torch.no_grad():
                all_sprt = {'imgs': [], 'masks': []}
                all_qry = {'imgs': [], 'masks': [], 'flows': [], 'paths': None}

                q_label = q_label.to(dist.get_rank(), non_blocking=True)
                spprt_imgs = spprt_imgs.to(dist.get_rank(), non_blocking=True)
                s_label = s_label.to(dist.get_rank(), non_blocking=True)
                qry_img = qry_img.to(dist.get_rank(), non_blocking=True)
                #np.save('qry.npy', qry_img.cpu().numpy())
                f_s = model.module.extract_features(spprt_imgs.squeeze(0))
                if qry_img.ndim > 4:
                    # B x N x C x H x W --> N x C X H X W
                    qry_img = qry_img.squeeze(0) # Squeeze batch dim
                    q_label = q_label.squeeze(0) # Squeeze batch dim
                f_q = model.module.extract_features(qry_img)

                Nframes = f_q.size(0)
                shot = f_s.size(0)

                iter_num += Nframes

                n_shots = torch.tensor([shot] * Nframes).to(dist.get_rank(), non_blocking=True)
                features_s = f_s.repeat(Nframes, 1, 1, 1, 1).detach()
                features_q = f_q.detach().unsqueeze(1)

                gt_s = s_label.repeat(Nframes, 1, 1, 1)
                gt_q = q_label.unsqueeze(1)
                classes = [class_.item() for class_ in subcls] * Nframes
                seqs = misc * Nframes

                if args.visu:
                    all_sprt['imgs'] = spprt_imgs.cpu().numpy()
                    all_sprt['masks'] = s_label.cpu().numpy()
                    all_qry['imgs'] = qry_img.cpu().numpy()
                    all_qry['masks'] = q_label.cpu().numpy()
                    all_qry['paths'] = os.path.join(val_loader.dataset.img_dir, misc[0])
                    all_qry['paths'] = [os.path.join(all_qry['paths'], fname) for fname in sorted(os.listdir(all_qry['paths'])) ]
                    if args.flow_aggregation:
                        all_qry['flows'] = qry_flow.cpu().numpy()


            # =========== Normalize features along channel dimension ===============
            if args.norm_feat:
                features_s = F.normalize(features_s, dim=2)
                features_q = F.normalize(features_q, dim=2)

            for method, weights in all_weights.items():
                # =========== Create a callback is args.visdom_port != -1 ===============
                callback = VisdomLogger(port=args.visdom_port, env=args.visdom_env) if use_callback else None

                # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
                classifier = Classifier(args)
                classifier.init_prototypes(features_s, features_q, gt_s, gt_q, classes, callback)
                batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)

                # =========== Perform RePRI inference ===============
                batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, seqs, callback,
                                                weights=weights, roi_grid=roi_grid, consistency_type=consistency_type)
                t1 = time.time()
                runtime[method] += t1 - t0
                logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w]
                logits = F.interpolate(logits,
                                       size=(H, W),
                                       mode='bilinear',
                                       align_corners=True)
                probas = classifier.get_probas(logits).detach()
                #np.save(f'{method}_probas.npy', probas.detach().cpu())
                intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2)  # [n_tasks, shot, num_class]
                intersection, union = intersection.cpu(), union.cpu()

                if args.tloss_type == 'temporal_repri' and args.enable_log:
                    log_temporal_repri(intersection, union, gt_q, features_q, probas, misc, method)

                if args.eval_vc:
                    video_consistency = batch_vid_consistencyGPU(seqs, probas, gt_q, 2)  # [n_tasks, shot, num_class]
                    video_consistency = video_consistency.cpu()
                else:
                    video_consistency = torch.zeros(len(seqs), 1, 2)

                # ================== Log metrics ==================
                one_hot_gt = to_one_hot(gt_q, 2)
                valid_pixels = gt_q != 255
                loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
                loss_meter.update(loss.item())

                visited_seqs = []
                for i, class_ in enumerate(classes):
                    cls_intersection[method][class_] += intersection[i, 0, 1]  # Do not count background
                    cls_union[method][class_] += union[i, 0, 1]
                    if seqs[i] not in visited_seqs:
                        visited_seqs.append(seqs[i])
                        cls_vc[method][class_] += video_consistency[i, 0, 1]
                        cls_n_vc[method][class_] += 1

                for class_ in cls_union[method]:
                    IoU[method][class_] = cls_intersection[method][class_] / (cls_union[method][class_] + 1e-10)
                    Fscores[method][class_] = 2 * cls_intersection[method][class_] / \
                                                (cls_union[method][class_] + cls_intersection[method][class_] + 1e-10)
                    cls_vc[method][class_] /= (cls_n_vc[method][class_] + 1e-10)

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
                        root = os.path.join('plots', 'episodes', method, 'split_%d'%args.train_split)
                        os.makedirs(root, exist_ok=True)
                        save_path = os.path.join(root, f'run_{run}_iter_{iter_num}_{i:05d}.png')
                        if args.flow_aggregation:
                            flow_q = all_qry['flows'][i]
                        else:
                            flow_q = None
                        make_episode_visualization_cv2(img_s=all_sprt['imgs'][0].copy(),
                                                       img_q=all_qry['imgs'][i].copy(),
                                                       gt_s=all_sprt['masks'][0].copy(),
                                                       gt_q=all_qry['masks'][i].copy(),
                                                       path_s=[all_qry['paths'][i]],
                                                       path_q=all_qry['paths'][i],
                                                       preds=probas[i].cpu().numpy().copy(),
                                                       save_path=save_path,
                                                       flow_q=flow_q)

        # ================== Evaluation Metrics on ALl episodes ==================
        for method in all_weights.keys():
            print('========= Method {}==========='.format(method))
            runtimes[method][run] = runtime[method]
            mIoU = np.mean(list(IoU[method].values()))
            fscore = np.mean(list(Fscores[method].values()))
            vc = np.mean(list(cls_vc[method].values()))
            print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
            for class_ in cls_union[method]:
                print("Class {} : {:.4f}".format(class_, IoU[method][class_]))

            val_Fscores[method][run] = fscore
            val_IoUs[method][run] = mIoU
            val_VCs[method][run] = vc
            val_losses[method][run] = loss_meter.avg

    # ================== Save metrics ==================
    for method in all_weights.keys():
        str_weights = str(all_weights[method])
        print(f'========Final Evaluation of {method} with weights {str_weights}============')
        print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs[method].mean()))
        print('Average Fscore over {} runs --- {:.4f}.'.format(args.n_runs, val_Fscores[method].mean()))
        print('Average Vconsistency over {} runs --- {:.4f}.'.format(args.n_runs, val_VCs[method].mean()))
        print('Average runtime / run --- {:.4f}.'.format(runtimes[method].mean()))

    # This method works on multiple weights can not be used outside
    return None, None


def standard_validate(args: argparse.Namespace,
                      val_loader: torch.utils.data.DataLoader,
                      model: DDP,
                      use_callback: bool,
                      suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Standard validation')
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    iterable_val_loader = iter(val_loader)

    bar = tqdm(range(len(iterable_val_loader)))

    loss = 0.
    intersections = torch.zeros(args.num_classes_tr).to(dist.get_rank())
    unions = torch.zeros(args.num_classes_tr).to(dist.get_rank())

    with torch.no_grad():
        for i in bar:
            images, gt = iterable_val_loader.next()
            images = images.to(dist.get_rank(), non_blocking=True)
            gt = gt.to(dist.get_rank(), non_blocking=True)

            if images.ndim > 4:
                # Flatten frames dim with batch
                images = images.view((-1, *images.shape[-3:]))
                gt = gt.view((-1, *gt.shape[-2:]))

            logits = model(images).detach()
            loss += loss_fn(logits, gt)
            intersection, union, _ = intersectionAndUnionGPU(logits.argmax(1),
                                                             gt,
                                                             args.num_classes_tr,
                                                             255)
            intersections += intersection
            unions += union
        loss /= len(val_loader.dataset)

    if args.distributed:
        dist.all_reduce(loss)
        dist.all_reduce(intersections)
        dist.all_reduce(unions)

    mIoU = (intersections / (unions + 1e-10)).mean()
    loss /= dist.get_world_size()
    return mIoU, loss


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    main_worker(0, world_size, args)
#    mp.spawn(main_worker,
#             args=(world_size, args),
#             nprocs=world_size,
#             join=True)
