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
from .model.model import get_model
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_train_loader, get_val_loader
from .util import intersectionAndUnionGPU, get_model_dir, AverageMeter, find_free_port
from .util import setup, cleanup, main_process
from .util import denorm, map_label
from tqdm import tqdm
from .test import standard_validate, episodic_validate
from .test_nonbatched import episodic_validate as temporal_episodic_validate
from typing import Dict
from torch import Tensor

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from typing import Tuple
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
from src.losses.contrastive_loss import dense_temporal_contrastive_loss
from src.util import init_or_resume_wandb_run
import wandb
import pathlib

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training')
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

    print(f"==> Running process rank {rank}.")
    setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    callback = None if args.visdom_port == -1 else VisdomLogger(port=args.visdom_port, env=args.visdom_env)

    # ========== Data  =====================
    train_loader, train_sampler = get_train_loader(args)
    val_loader, _ = get_val_loader(args)  # mode='train' means that we will validate on images from validation set, but with the bases classes

    # ========== Model + Optimizer ==========
    model = get_model(args).to(rank)
    modules_ori = model.get_backbone_modules()
    modules_new = model.get_new_modules()

    if hasattr(model, 'cl_proj_head'):
        modules_new += [model.cl_proj_head]

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
    optimizer = get_optimizer(args, params_list)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # DDP DOes not work with torch.no_grad in FOrward which is used in HSNet
    find_unused = False
    if hasattr(args, 'model_type') and args.model_type=='hsnet':
        find_unused = True
    elif args.arch == "videoswin":
        find_unused = True

    model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused)

    # ========== Validation ==================
    validate_fn = episodic_validate if args.episodic_val else standard_validate
    validate_fn = temporal_episodic_validate if args.temporal_episodic_val == 4 else validate_fn
    #validate_fn = temporal_episodic_validate_meta if args.temporal_episodic_val == 3 else validate_fn

    # ========== Train Fn ==================
    # train_fn = do_episodic_epoch if hasattr(args, 'episodic_train') and args.episodic_train else do_epoch
    train_fn = do_epoch

    # ============ Pretrain model if needed =================
    if hasattr(args, 'pretrained_path'):
        original_type = args.model_type
        pretrained_type = args.model_type if not hasattr(args, 'pretrained_type') else args.pretrained_type
        args.model_type = pretrained_type
        savedir = get_model_dir(args, ckpt_path=args.pretrained_path)
        args.model_type = original_type
        model_path = os.path.join(savedir, 'best.pth')
        print('Pretraining from ', model_path)
        loaded_model = torch.load(model_path)
        loaded_model_final = {k: v for k, v in loaded_model['state_dict'].items() if 'classifier' not in k}
        model.load_state_dict(loaded_model_final, strict=False)

    # ============ Resume model if exists =================
    savedir = get_model_dir(args)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    best_model_path = os.path.join(savedir, 'best.pth')
    if os.path.exists(best_model_path) and args.resume_training:
        print('Resuming Training from ', best_model_path)
        best_model = torch.load(best_model_path)
        model.load_state_dict(best_model['state_dict'])
        optimizer.load_state_dict(best_model['optimizer'])
        if args.epochs > best_model['epoch']:
            args.epochs = args.epochs - best_model['epoch']
        if train_sampler is not None:
            train_sampler.set_epoch(best_model['epoch'])

    # =========================== Setup WANDB ====================
    if args.wandb_run_name != '' and main_process(args):
        wandb_id_file_path = pathlib.Path(savedir + '/' + args.wandb_run_name + '.txt')
        config = init_or_resume_wandb_run(wandb_id_file_path,
                                          entity_name=args.wandb_user,
                                          project_name=args.wandb_project,
                                          run_name=args.wandb_run_name,
                                          config=args)

    # ========== Scheduler  ================
    scheduler = get_scheduler(args, optimizer, len(train_loader))

    # ========== Metrics initialization ====
    max_val_mIoU = 0.
    if args.debug:
        iter_per_epoch = 5
    else:
        iter_per_epoch = len(train_loader)
    log_iter = int(iter_per_epoch / args.log_freq) + 1

    metrics: Dict[str, Tensor] = {"val_mIou": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "val_loss": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "train_mIou": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  "train_loss": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  }

    # ========== Training  =================
    for epoch in tqdm(range(args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_mIou, train_loss = train_fn(args=args,
                                          train_loader=train_loader,
                                          iter_per_epoch=iter_per_epoch,
                                          model=model,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          epoch=epoch,
                                          callback=callback,
                                          log_iter=log_iter)

        val_mIou, val_loss = validate_fn(args=args,
                                         val_loader=val_loader,
                                         model=model,
                                         use_callback=False,
                                         suffix=f'train_{epoch}')

        if args.distributed:
            dist.all_reduce(val_mIou), dist.all_reduce(val_loss)
            val_mIou /= world_size
            val_loss /= world_size

        if main_process(args):
            # Live plot if desired with visdom
            if callback is not None:
                callback.scalar('val_loss', epoch, val_loss, title='Validiation Loss')
                callback.scalar('mIoU_val', epoch, val_mIou, title='Val metrics')

            elif args.wandb_run_name != '':
                wandb.log({'val_loss': val_loss,
                           'mIoU_val': val_mIou})

            # Model selection
            if val_mIou.item() > max_val_mIoU:
                max_val_mIoU = val_mIou.item()
                os.makedirs(savedir, exist_ok=True)
                filename = os.path.join(savedir, f'best.pth')
                if args.save_models:
                    print('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'args': args}, filename)
            print("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

            # Sort and save the metrics
            for k in metrics:
                metrics[k][epoch] = eval(k)

            for k, e in metrics.items():
                path = os.path.join(savedir, f"{k}.npy")
                np.save(path, e.cpu().numpy())

    if args.save_models and main_process(args):
        filename = os.path.join(savedir, 'final.pth')
        print(f'Saving checkpoint to: {filename}')
        torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'args': args}, filename)

    cleanup()
    wandb.finish()

def iou_loss(preds: torch.tensor,
             one_hot: torch.tensor,
             targets: torch.tensor,
             ignore_index: int = 255) -> torch.tensor:
    """
    preds: predictions 0->1, [Bx C x H x W]
    one_hot: one hot vector masks [B x C x H x W]
    targets: has the original groundtruth with ignore index [B x H x W]
    """
    # use this as reference: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    Iand1 = (one_hot * preds).permute(1,0,2,3)
    Ior1 = (one_hot + preds).permute(1,0,2,3) - Iand1

    IoU = torch.mean(Iand1[:, targets!=ignore_index] / (Ior1[:, targets!=ignore_index] + 1e-10), dim=1)
    return (1 - IoU).mean()

def cross_entropy(logits: torch.tensor,
                  one_hot: torch.tensor,
                  targets: torch.tensor,
                  mean_reduce: bool = True,
                  ignore_index: int = 255) -> torch.tensor:

    """
    inputs:
        one_hot  : shape [batch_size, num_classes, h, w]
        logits : shape [batch_size, num_classes, h, w]
        targets : shape [batch_size, h, w]
    returns:
        loss: shape [batch_size] or [] depending on mean_reduce

    """
    assert logits.size() == one_hot.size()
    log_prb = F.log_softmax(logits, dim=1)
    non_pad_mask = targets.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask)
    if mean_reduce:
        return loss.mean()  # average later
    else:
        return loss


def compute_loss(args: argparse.Namespace,
                 model: DDP,
                 images: torch.tensor,
                 targets: torch.tensor,
                 num_classes: int,
                 aux_images: torch.tensor = None,
                 aux_gt: torch.tensor = None) -> torch.tensor:
    """
    inputs:
        images  : shape [batch_size, C, h, w]
        logits : shape [batch_size, num_classes, h, w]
        targets : shape [batch_size, h, w]
        aux_images: shape [batch_size, C, h, w]
        aux_gt: shape [batch_size, h, w] Only has [0, 255] to highlight invalid pixels from padding
    returns:
        loss: shape []
        logits: shape [batch_size]

    """
    loss_dict = {}
    loss = torch.tensor([0]).float().to(dist.get_rank())
    batch, h, w = targets.size()
    one_hot_mask = torch.zeros(batch, num_classes, h, w).to(dist.get_rank())
    new_target = targets.clone().unsqueeze(1)
    new_target[new_target == 255] = 0

    one_hot_mask.scatter_(1, new_target, 1).long()
    if args.smoothing:
        eps = 0.1
        one_hot = one_hot_mask * (1 - eps) + (1 - one_hot_mask) * eps / (num_classes - 1)
    else:
        one_hot = one_hot_mask  # [batch_size, num_classes, h, w]

    loss_type = 'ce'
    if hasattr(args, 'training_loss'):
        loss_type = args.training_loss

    if loss_type == 'iou':
        # IoU Loss
        logits = model(images)
        preds = F.softmax(logits, dim=1)
        loss_dict['IOU'] = iou_loss(preds, one_hot, targets)
        loss += loss_dict['IOU']

    elif loss_type == 'ce_iou':
        logits = model(images)
        loss_dict['CE']  = cross_entropy(logits, one_hot, targets)
        preds = F.softmax(logits, dim=1)
        loss_dict['IOU'] = iou_loss(preds[:, 1:], one_hot[:, 1:], targets)
        loss = loss_dict['CE'] + loss_dict['IOU']

    else:
        # Cross entropy loss, can have auxiliary contrastive
        if args.mixup:
            assert args.model_type != 'fpn', "mixup not supported for fpn"
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(images.size()[0]).to(dist.get_rank())
            one_hot_a = one_hot
            targets_a = targets

            one_hot_b = one_hot[rand_index]
            target_b = targets[rand_index]
            mixed_images = lam * images + (1 - lam) * images[rand_index]

            logits = model(mixed_images)
            loss_dict['CE']  = cross_entropy(logits, one_hot_a, targets_a) * lam  \
                   + cross_entropy(logits, one_hot_b, target_b) * (1. - lam)
        else:
            logits = model(images)
            if type(logits) == list:
                # Res FPN return list of outputs
                ce = 0
                for l in logits:
                    ce += cross_entropy(l, one_hot, targets)
                loss_dict['CE'] = ce
            else:
                loss_dict['CE']  = cross_entropy(logits, one_hot, targets)

        loss += loss_dict['CE']

        if aux_images is not None:
            assert len(torch.unique(aux_gt)) <= 2, 'Wrongly Labelled Auxiliary'
            if len(torch.unique(aux_gt)) == 2:
                assert 255 in torch.unique(aux_gt), 'Wrongly Labelled Auxiliary'

            # Compute contrastive Loss
            aux_images = aux_images.view(-1, *aux_images.shape[-3:])
            _, aux_features = model(aux_images, interm=args.densecl_interm, avg_pool=args.densecl_avgpool,
                                    projection=args.densecl_proj)

            aux_features = aux_features.view(-1, args.aux_temporal_window, \
                                             *aux_features.shape[-3:])
            loss_dict['TCL'] = dense_temporal_contrastive_loss(aux_features, temperature=args.densecl_temperature,
                                                               rank=dist.get_rank(), aux_gt=aux_gt)
            loss += args.densecl_lamda * loss_dict['TCL']

    return loss, loss_dict

def do_epoch(args: argparse.Namespace,
             train_loader: torch.utils.data.DataLoader,
             model: DDP,
             optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler,
             epoch: int,
             callback: VisdomLogger,
             iter_per_epoch: int,
             log_iter: int) -> Tuple[torch.tensor, torch.tensor]:
    loss_meter = AverageMeter()
    loss_dict_meter = {}
    train_vis_dict = {}

    train_losses = torch.zeros(log_iter).to(dist.get_rank())
    train_mIous = torch.zeros(log_iter).to(dist.get_rank())

    iterable_train_loader = iter(train_loader)

    if main_process(args):
        bar = tqdm(range(iter_per_epoch))
    else:
        bar = range(iter_per_epoch)

    for i in bar:
        model.train()
        current_iter = epoch * len(train_loader) + i + 1

        # =========== Prepare Input ================
        images, gt, aux_flag = iterable_train_loader.next()
        if type(aux_flag) != int:
            aux_flag = aux_flag[0]

        if aux_flag:
            aux_images = images["aux_images"]
            images = images["images"]

            aux_gt = gt["aux_labels"]
            gt = gt["labels"]
        else:
            aux_images = None
            aux_gt = None

        if aux_images is not None:
            aux_images = aux_images.to(dist.get_rank(), non_blocking=True)
            aux_gt = aux_gt.to(dist.get_rank(), non_blocking=True)

        images = images.to(dist.get_rank(), non_blocking=True)
        gt = gt.to(dist.get_rank(), non_blocking=True)

        if images.ndim > 4:
            # Flatten frames dim with batch
            if args.arch != "videoswin":
                images = images.view((-1, *images.shape[-3:]))
            gt = gt.view((-1, *gt.shape[-2:])).long()

        # ============ Compute Loss =================
        loss, loss_dict = compute_loss(args=args,
                                       model=model,
                                       images=images,
                                       targets=gt,
                                       num_classes=args.num_classes_tr,
                                       aux_images=aux_images,
                                       aux_gt=aux_gt)

        # ===================== Optimization ================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.scheduler == 'cosine':
            scheduler.step()

        # ==================== Logging ====================
        if i % args.log_freq == 0:
            if not hasattr(args, 'pretrain_cl') or not args.pretrain_cl:
                model.eval()
                logits = model(images)
                intersection, union, target = intersectionAndUnionGPU(logits.argmax(1),
                                                                      gt,
                                                                      args.num_classes_tr,
                                                                      255)
                images = images.cpu()
                logits = logits.cpu()
                gt = gt.cpu()

                if args.distributed:
                    dist.all_reduce(loss)
                    dist.all_reduce(intersection)
                    dist.all_reduce(union)
                    dist.all_reduce(target)

                allAcc = (intersection.sum() / (target.sum() + 1e-10))  # scalar
                mAcc = (intersection / (target + 1e-10)).mean()
                mIoU = (intersection / (union + 1e-10)).mean()
            else:
                allAcc = 0
                mAcc = 0
                mIoU = 0

            loss_meter.update(loss.item() / dist.get_world_size())
            for k, v in loss_dict.items():
                if k not in loss_dict_meter:
                    loss_dict_meter[k] = AverageMeter()
                loss_dict_meter[k].update(v.item() / dist.get_world_size())

            if main_process(args):
                if args.arch == "videoswin":
                    # Reshape Images with temporal and batch to be same dim for visualisation
                    images = images.view((-1, *images.shape[-3:]))

                if callback is not None:
                    t = current_iter / len(train_loader)
                    callback.scalar('loss_train_batch', t, loss_meter.avg, title='Loss')
                    for k, v in loss_dict_meter.items():
                        callback.scalar('loss_train_batch_%s'%k, t, loss_dict_meter[k].avg, title='Separate Loss')

                    callback.scalars(['mIoU', 'mAcc', 'allAcc'], t,
                                     [mIoU, mAcc, allAcc],
                                     title='Training metrics')
                    for index, param_group in enumerate(optimizer.param_groups):
                        lr = param_group['lr']
                        callback.scalar('lr', t, lr, title='Learning rate')
                        break

                elif args.wandb_run_name != '':
                    for index, param_group in enumerate(optimizer.param_groups):
                        train_vis_dict['lr'] = param_group['lr']
                        break

                    train_vis_dict['loss_train_batch'] = loss_meter.avg
                    for k, v in loss_dict_meter.items():
                        train_vis_dict['loss_train_batch_%s'%k] = loss_dict_meter[k].avg

                denorm_img = denorm(images[0].cpu())
                if not hasattr(args, 'pretrain_cl') or not args.pretrain_cl:
                    lbl, colors = map_label(gt[0].cpu(), nclasses=args.num_classes_tr)
                    pred, _ = map_label(torch.argmax(logits[0].cpu().detach(), dim=0), colors=colors)
                    concat_img = np.expand_dims(np.concatenate((denorm_img, lbl, pred), axis=2), axis=0)
                    if callback is not None:
                        callback.images('Train Sample', images=concat_img, title="Train Image Labels")

                    elif args.wandb_run_name != '':
                        train_vis_dict['Train Sample'] = wandb.Image(concat_img[0].transpose(1,2,0))

                    if aux_images is not None:
                        # No labels or predictions exist
                        image_pairs = aux_images.view(-1, args.aux_temporal_window, *aux_images.shape[-3:])
                        concat_img_pairs = []
                        for i in range(image_pairs.shape[0]):
                            concat_img_pairs.append(np.concatenate((denorm(image_pairs[i, 0].cpu()),
                                                                   denorm(image_pairs[i, 1].cpu())), axis=1))
                        concat_img_pairs = np.stack(concat_img_pairs)
                        callback.images('Train Sample Aux', images=concat_img_pairs, title="Train Image Pairs, Positive Negative")
                else:
                    # No labels or predictions exist
                    image_pairs = images.view(-1, args.aux_temporal_window, *images.shape[-3:])
                    concat_img_pairs = []
                    for i in range(image_pairs.shape[0]):
                        concat_img_pairs.append(np.concatenate((denorm(image_pairs[i, 0].cpu()),
                                                               denorm(image_pairs[i, 1].cpu())), axis=1))
                    concat_img_pairs = np.stack(concat_img_pairs)
                    callback.images('Train Sample', images=concat_img_pairs, title="Train Image Pairs, Positive Negative")

                train_losses[int(i / args.log_freq)] = loss_meter.avg
                train_mIous[int(i / args.log_freq)] = mIoU

                if args.wandb_run_name != '':
                    wandb.log(train_vis_dict)

    if args.scheduler != 'cosine':
        scheduler.step()

    if aux_images is not None:
        train_loader.dataset.reset_indices()

    return train_mIous, train_losses


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        # args.epochs = 2
        args.n_runs = 2
        args.save_models = False

    world_size = len(args.gpus)
    distributed = world_size > 1
    args.distributed = distributed
    args.port = find_free_port()
    if args.debug:
        # In debugging mode don use multile processes
        main_worker(0, world_size, args)
    else:
        try:
            mp.spawn(main_worker,
                     args=(world_size, args),
                     nprocs=world_size,
                     join=True)
        except KeyboardInterrupt:
            print('Interrupted')
            print('Killing processes Explicitly')

            cleanup()
            wandb.finish()

            #os.system("kill $(ps ux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
