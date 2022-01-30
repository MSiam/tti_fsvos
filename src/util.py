import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml
import copy
from ast import literal_eval
from typing import Callable, Iterable, List, TypeVar, Optional, Dict
import torch.distributed as dist
from typing import Tuple
import argparse
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion
import wandb
import pathlib


def get_split_base_protos(self, class_mapping):
    base_protos = np.load('/local/data0/avg_protos.npy', allow_pickle=True).item()
    base_protos = base_protos['res5']
    protos = {}

    class_mapping = [0] + class_mapping
    for cls_idx, cls in enumerate(class_mapping):
        protos[cls_idx] = base_protos[cls]
    assert len(protos.keys()) == 31
    return protos

def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             entity_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.
        Returns the config, if it's not None it will also update it first
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        print('Resuming from wandb path... ', wandb_id_file_path)
        resume_id = wandb_id_file_path.read_text()
        wandb.init(entity=entity_name,
                   project=project_name,
                   name=run_name,
                   resume=resume_id,
                   config=config)
                   # settings=wandb.Settings(start_method="thread"))
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(entity=entity_name, project=project_name, name=run_name, config=config)
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        wandb.config.update(config)

    return config

A = TypeVar("A")
B = TypeVar("B")

def create_pseudogt(proba):
    proba_temp = torch.argmax(proba, dim=2)
    proba = torch.ones_like(proba_temp) * 255

    # Create distance transform to create pseudo gt with ignore pixels around boundary
    # To avoid propagation of errors
    erosion_size = 3
    pos_th = 0.8
    h, w = proba.shape[-2:]
    neg_th = 0.2 * np.sqrt(h**2+w**2)

    #eroded_mask = grey_erosion(proba_temp.cpu(), size=(1,1,erosion_size, erosion_size))
    # Compute distance transform
    #dt = distance_transform_edt(np.logical_not(eroded_mask))
    dt = distance_transform_edt(np.logical_not(proba_temp.cpu() ))

    negatives = torch.tensor(dt > neg_th)
    proba[proba_temp==1] = 1
    proba[negatives] = 0
    return proba

def get_interval(grid, roi_window):
    grid = grid[:-1]
    if grid == 'c':
        start = - roi_window // 2
        end = roi_window // 2
    elif grid == 'min':
        start = 0
        end = roi_window
    elif grid == 'max':
        start = - roi_window
        end = 0
    return start, end

def generate_roi_grid(h, w, roi_window=10, grids=['cx-cy']):
    # cx: centerx, cy: centery, minx: minimum x, maxx: maximum x, ..

    rois = []
    for grid in grids:
        gridx, gridy = grid.split('-')
        roi = torch.zeros((h*w, h*w))
        for y in range(h):
            for x in range(w):
                cidx = y*w+x

                startx, endx = get_interval(gridx, roi_window)
                starty, endy = get_interval(gridy, roi_window)

                for ky in range(starty, endy):
                    for kx in range(startx, endx):

                        iky = y+ky if y+ky > 0 else 0
                        iky = iky if iky < h else h-1
                        ikx = x+kx if x+kx > 0 else 0
                        ikx = ikx if ikx < w else w-1

                        kidx = iky*w + ikx
                        roi[cidx, kidx] = 1
        rois.append(roi.cuda())
    return rois

def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(args: argparse.Namespace,
          rank: int,
          world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """
    Used for multiprocessing
    """
    return list(map(fn, iter))


def get_model_dir(args: argparse.Namespace, ckpt_path: str='') -> str:
    """
    Obtain the directory to save/load the model
    """
    if ckpt_path == '':
        ckpt_path = args.ckpt_path

    model_type = 'pspnet'
    if hasattr(args, 'model_type'):
        model_type = args.model_type

    path = os.path.join(ckpt_path,
                        args.train_name,
                        f'split={args.train_split}',
                        'model',
                        f'{model_type}_{args.arch}{args.layers}',
                        f'smoothing={args.smoothing}',
                        f'mixup={args.mixup}')
    return path

def denorm(img, mean=None, scale=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
        scale = [0.229, 0.224, 0.225]

    img = img.permute(1,2,0)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.permute(2, 0, 1).cpu().numpy()
    img = np.asarray(img*255, np.uint8)
    return img

def map_label(lbl, colors=None, nclasses=None):
    if colors is None:
        colors = {i+1: np.random.random((1,3))*255 for i in range(nclasses)}
    colors[255] = np.ones((1,3))*255

    colored_lbl = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    for cls in np.unique(lbl):
        if cls == 0:
            continue
        colored_lbl[lbl==cls] = colors[cls]
    return colored_lbl.transpose(2,0,1), colors

def to_one_hot(mask: torch.tensor,
               num_classes: int) -> torch.tensor:
    """
    inputs:
        mask : shape [n_task, shot, h, w]
        num_classes : Number of classes

    returns :
        one_hot_mask : shape [n_task, shot, num_class, h, w]
    """
    n_tasks, shot, h, w = mask.size()
    one_hot_mask = torch.zeros(n_tasks, shot, num_classes, h, w).to(dist.get_rank())
    new_mask = mask.unsqueeze(2).clone()
    new_mask[torch.where(new_mask == 255)] = 0  # Ignore_pixels are anyways filtered out in the losses
    one_hot_mask.scatter_(2, new_mask, 1).long()
    return one_hot_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def vid_consistencyGPU(seqs: List[str],
                       preds: torch.Tensor,
                       target: torch.Tensor,
                       num_classes: int,
                       size_th: int = 225, # 15**2
                       window: int = 3,
                       ignore_index=255) -> torch.tensor:

    assert num_classes == 2, "VC does not support more than 2 classes including background"
    seqs = np.array(seqs)
    n_tasks = seqs.shape[0]
    video_consistency = torch.zeros(n_tasks, num_classes).cuda()
    ignored_seqs = []

    clip_length = window

    assert preds.shape == target.shape

    seqs_unique = np.unique(seqs)
    # Go over all sequences in this bath
    for seq in seqs_unique:
        seq_indices = np.where(seqs==seq)[0]
        current_vc = []
        # Go over all frames
        for i in range(len(seq_indices)-clip_length):
            common_gt_area = torch.ones(preds.shape[-2:]).cuda()
            common_pred_area = torch.ones(preds.shape[-2:]).cuda()

            previous_pred = None
            previous_target = None

            # Go through clip surrounding frame with +3
            for j in range(clip_length):
                current_pred = preds[i+j]
                current_target = target[i+j]

                # Background is always ignored too in VC computation
                current_pred[current_target==ignore_index] = 0
                current_target[current_target==ignore_index] = 0

                if previous_pred is not None and previous_target is not None:
                    common_gt_area = (current_target * previous_target) * common_gt_area
                    common_pred_area = (current_pred * previous_pred) * common_pred_area

                previous_pred = current_pred
                previous_target = current_target

#            if common_gt_area.sum() < size_th:
#                continue

            current_vc.append((common_gt_area * common_pred_area).sum() / (common_gt_area.sum() + 1e-10) )

        if len(current_vc) != 0: # skip small seqs in frames or with small object sizes
            video_consistency[seq_indices, 1] = torch.stack(current_vc).mean()
        else:
            ignored_seqs.append(seq)

    return video_consistency

def batch_vid_consistencyGPU(seqs: List[str],
                             logits: torch.Tensor,
                             target: torch.Tensor,
                             num_classes: int,
                             size_th: int = 225, # 15**2
                             windows: List[int] = [3],
                             ignore_index=255,
                             ) -> torch.tensor:
    """
    inputs:
        logits : shape [n_task, shot, num_class, h, w]
        target : shape [n_task, shot, H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [n_task, shot, num_class]
        area_union : shape [n_task, shot, num_class]
        area_target : shape [n_task, shot, num_class]
    """


    n_tasks, shots, num_classes, h, w = logits.size()
    H, W = target.size()[-2:]

    logits = F.interpolate(logits.view(n_tasks * shots, num_classes, h, w),
                           size=(H, W), mode='bilinear', align_corners=True).view(n_tasks, shots, num_classes, H, W)
    preds = logits.argmax(2)  # [n_task, shot, H, W]

    video_consistency = {}
    for window in windows:
        video_consistency[window] = torch.zeros(n_tasks, shots, num_classes)

        for shot in range(shots):
            vc = vid_consistencyGPU(seqs, preds[:,shot], target[:,shot],
                                    num_classes, size_th=size_th, window=window,
                                    ignore_index=ignore_index)

            video_consistency[window][:, shot, :] = vc

    return video_consistency

def batch_intersectionAndUnionGPU(logits: torch.Tensor,
                                  target: torch.Tensor,
                                  num_classes: int,
                                  ignore_index=255) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        logits : shape [n_task, shot, num_class, h, w]
        target : shape [n_task, shot, H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [n_task, shot, num_class]
        area_union : shape [n_task, shot, num_class]
        area_target : shape [n_task, shot, num_class]
    """
    n_task, shots, num_classes, h, w = logits.size()
    H, W = target.size()[-2:]

    logits = F.interpolate(logits.view(n_task * shots, num_classes, h, w),
                           size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots, num_classes, H, W)
    preds = logits.argmax(2)  # [n_task, shot, H, W]

    n_tasks, shot, num_classes, H, W = logits.size()
    area_intersection = torch.zeros(n_tasks, shot, num_classes)
    area_union = torch.zeros(n_tasks, shot, num_classes)
    area_target = torch.zeros(n_tasks, shot, num_classes)
    for task in range(n_tasks):
        for shot in range(shots):
            i, u, t = intersectionAndUnionGPU(preds[task][shot], target[task][shot],
                                              num_classes, ignore_index=ignore_index) # i,u, t are of size()
            area_intersection[task, shot, :] = i
            area_union[task, shot, :] = u
            area_target[task, shot, :] = t
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(preds: torch.tensor,
                            target: torch.tensor,
                            num_classes: int,
                            ignore_index=255) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    inputs:
        preds : shape [H, W]
        target : shape [H, W]
        num_classes : Number of classes

    returns :
        area_intersection : shape [num_class]
        area_union : shape [num_class]
        area_target : shape [num_class]
    """
    assert (preds.dim() in [1, 2, 3])
    assert preds.shape == target.shape
    preds = preds.view(-1)
    target = target.view(-1)
    preds[target == ignore_index] = ignore_index
    intersection = preds[preds == target]

    # Addind .float() becausue histc not working with long() on CPU
    area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
    area_output = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_output + area_target - area_intersection
    # print(torch.unique(intersection))
    return area_intersection, area_union, area_target

def compute_map(mask, val_pixels, features, collapse_shot_dim=True):
    """
    mask: batch x shot x 2 x H x W
    val_pixels: batch x shot x H x W
    features: batch x shot x C x H x W
    """
    mask = mask[:, :, 1:] # Access Foreground Mask
    mask = mask * val_pixels.unsqueeze(2)
    if collapse_shot_dim:
        protos = (features * mask).sum(dim=(1, 3, 4))
        protos /= (mask.sum(dim=(1, 3, 4)) + 1e-10)
        protos = F.normalize(protos, dim=1)
    else:
        protos = (features * mask).sum(dim=(3, 4))
        protos /= (mask.sum(dim=(3, 4)) + 1e-10)
        protos = F.normalize(protos, dim=2)

    return protos

# ======================================================================================================================
# ======== All following helper functions have been borrowed from from https://github.com/Jia-Research-Lab/PFENet ======
# ======================================================================================================================

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg
