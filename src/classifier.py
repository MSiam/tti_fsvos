import os
import torch
import torch.nn.functional as F
from src.util import batch_intersectionAndUnionGPU
from typing import List
from .util import to_one_hot, compute_map, create_pseudogt
from collections import defaultdict
from typing import Tuple
from visdom_logger import VisdomLogger
import numpy as np
from src.losses.temporal_losses import temporal_positive_steered, temporal_negative_steered_spatial
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion
import cv2

class Classifier(object):
    def __init__(self, args):
        self.num_classes = 2
        self.temperature = args.temperature
        self.adapt_iter = args.adapt_iter
        self.weights = args.weights
        self.lr = args.cls_lr
        self.FB_param_update = args.FB_param_update
        self.visdom_freq = args.cls_visdom_freq
        self.FB_param_type = args.FB_param_type
        self.FB_param_noise = args.FB_param_noise

        self.single_proto_flag = args.single_proto_flag if hasattr(args, 'single_proto_flag') else False

        if hasattr(args, 'temporal_window'):
            self.temporal_window = args.temporal_window
        else:
            self.temporal_window = 3

        if hasattr(args, 'temporal_step'):
            if args.temporal_step == -1:
                self.temporal_step = self.temporal_window
            else:
                self.temporal_step = args.temporal_step
        else:
            self.temporal_step = 1

        self.keyframe_criteria = 'keyframe_sprt'
        if hasattr(args, 'keyframe_criteria'):
            self.keyframe_criteria = args.keyframe_criteria

        if hasattr(args, 'nviews'):
            self.nviews = args.nviews

        if hasattr(args, 'tloss_type'):
            self.tloss_type = args.tloss_type
        else:
            self.tloss_type = None

        self.enable_temporal = False
        self.refine_iter = args.refine_iter if hasattr(args, 'refine_iter') else -1

    def init_prototypes(self, features_s: torch.tensor, features_q: torch.tensor,
                        gt_s: torch.tensor, gt_q: torch.tensor, subcls: List[int],
                        callback, seqs: List[str]) -> None:
        """
        inputs:
            features_s : shape [n_task, shot, c, h, w]
            features_q : shape [n_task, 1, c, h, w]
            gt_s : shape [n_task, shot, H, W]
            gt_q : shape [n_task, 1, H, W]

        returns :
            prototypes : shape [n_task, c]
            bias : shape [n_task]
        """

        # DownSample support masks
        n_task, shot, c, h, w = features_s.size()
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
        ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_task, shot, 1, h, w]

        # Computing prototypes
        fg_mask = (ds_gt_s == 1)
        fg_prototype = (features_s * fg_mask).sum(dim=(1, 3, 4))
        fg_prototype /= (fg_mask.sum(dim=(1, 3, 4)) + 1e-10)  # [n_task, c]
        if self.single_proto_flag:
            unique_seqs = np.unique(seqs)
            n_seqs = len(unique_seqs)
            self.prototype = []
            for seq in unique_seqs:
                self.prototype.append(
                        fg_prototype[np.where(seqs==seq)[0][0]]
                )
            self.prototype = torch.stack(self.prototype)
            proto_size = n_seqs
        else:
            self.prototype = fg_prototype
            proto_size = n_task

        logits_q = self.get_logits(features_q, seqs=seqs)  # [n_tasks, shot, h, w]
        self.bias = logits_q.mean(dim=(1, 2, 3))

        assert self.prototype.size() == (proto_size, c), self.prototype.size()
        assert torch.isnan(self.prototype).sum() == 0, self.prototype

        if callback is not None:
            self.update_callback(callback, 0, features_s, features_q, subcls, gt_s, gt_q, seqs=seqs)

    def get_logits(self, features: torch.tensor, ext_prototype: torch.tensor=None,
            seqs: List[str]=None, selected_seq: str = None) -> torch.tensor:

        """
        Computes the cosine similarity between self.prototype and given features
        inputs:
            features : shape [n_tasks, shot, c, h, w]

        returns :
            logits : shape [n_tasks, shot, h, w]
        """

        # Put prototypes and features in the right shape for multiplication
        features = features.permute((0, 1, 3, 4, 2))  # [n_task, shot, h, w, c]
        if self.single_proto_flag:
            # Broadcast unique prototypes to different seq frames
            unique_seqs = np.unique(seqs)
            prototype = torch.zeros(features.shape[0], 1, 1, features.shape[-1]).cuda()
            for i, seq in enumerate(unique_seqs):
                prototype[np.where(seqs==seq)[0]] = self.prototype[i]
        else:
            if ext_prototype is None:
                prototype = self.prototype.unsqueeze(1).unsqueeze(2)  # [n_tasks, 1, 1, c]
                if selected_seq is not None:
                    prototype = prototype[seqs == selected_seq]

            else:
                # Use it for temporal sequence same sprt set over all frames
                prototype = ext_prototype.unsqueeze(1).unsqueeze(2)  # [n_tasks, 1, 1, c]

        # Compute cosine similarity
        cossim = features.matmul(prototype.unsqueeze(4)).squeeze(4)  # [n_task, shot, h, w]
        cossim /= ((prototype.unsqueeze(3).norm(dim=4) * \
                    features.norm(dim=4)) + 1e-10)  # [n_tasks, shot, h, w]

        return self.temperature * cossim

    def get_probas(self, logits: torch.tensor, seqs: List[str]=None, selected_seq: str=None) -> torch.tensor:
        """
        inputs:
            logits : shape [n_tasks, shot, h, w]

        returns :
            probas : shape [n_tasks, shot, num_classes, h, w]
        """
        bias = self.bias.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        if selected_seq is not None:
            bias = bias[seqs == selected_seq]

        logits_fg = logits - bias # [n_tasks, shot, h, w]
        probas_fg = torch.sigmoid(logits_fg).unsqueeze(2)
        probas_bg = 1 - probas_fg
        probas = torch.cat([probas_bg, probas_fg], dim=2)
        return probas

    def compute_FB_param(self, features_q: torch.tensor, gt_q: torch.tensor, seqs: List[str]) -> torch.tensor:
        """
        inputs:
            features_q : shape [n_tasks, shot, c, h, w]
            gt_q : shape [n_tasks, shot, h, w]

        updates :
             self.FB_param : shape [n_tasks, num_classes]
        """
        ds_gt_q = F.interpolate(gt_q.float(), size=features_q.size()[-2:], mode='nearest').long()
        valid_pixels = (ds_gt_q != 255).unsqueeze(2)  # [n_tasks, shot, num_classes, h, w]
        assert (valid_pixels.sum(dim=(1, 2, 3, 4)) == 0).sum() == 0, valid_pixels.sum(dim=(1, 2, 3, 4))

        one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [n_tasks, shot, num_classes, h, w]

        oracle_FB_param = (valid_pixels * one_hot_gt_q).sum(dim=(1, 3, 4)) / valid_pixels.sum(dim=(1, 3, 4))

        if self.FB_param_type == 'oracle':
            self.FB_param = oracle_FB_param
            # Used to assess influence of delta perturbation
            if self.FB_param_noise != 0:
                perturbed_FB_param = oracle_FB_param
                perturbed_FB_param[:, 1] += self.FB_param_noise * perturbed_FB_param[:, 1]
                perturbed_FB_param = torch.clamp(perturbed_FB_param, 0, 1)
                perturbed_FB_param[:, 0] = 1.0 - perturbed_FB_param[:, 1]
                self.FB_param = perturbed_FB_param

        else:
            logits_q = self.get_logits(features_q, seqs=seqs)
            probas = self.get_probas(logits_q).detach()

            self.FB_param = (valid_pixels * probas).sum(dim=(1, 3, 4))
            self.FB_param /= valid_pixels.sum(dim=(1, 3, 4))

        # Compute the relative error
        deltas = self.FB_param[:, 1] / oracle_FB_param[:, 1] - 1
        return deltas

    def get_entropies(self,
                      valid_pixels: torch.tensor,
                      probas: torch.tensor,
                      reduction='sum') -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        inputs:
            probas : shape [n_tasks, shot, num_class, h, w]
            valid_pixels: shape [n_tasks, shot, h, w]

        returns:
            d_kl : FB proportion kl [n_tasks,]
            cond_entropy : Entropy of predictions [n_tasks,]
            marginal : Current marginal distribution over labels [n_tasks, num_classes]
        """
        n_tasks, shot, num_classes, h, w = probas.size()
        assert (valid_pixels.sum(dim=(1, 2, 3)) == 0).sum() == 0, \
               (valid_pixels.sum(dim=(1, 2, 3)) == 0).sum()  # Make sure all tasks have a least 1 valid pixel

        cond_entropy = - ((valid_pixels.unsqueeze(2) * (probas * torch.log(probas + 1e-10))).sum(2))
        cond_entropy = cond_entropy.sum(dim=(1, 2, 3))
        cond_entropy /= valid_pixels.sum(dim=(1, 2, 3))

        marginal = (valid_pixels.unsqueeze(2) * probas).sum(dim=(1, 3, 4))
        marginal /= valid_pixels.sum(dim=(1, 2, 3)).unsqueeze(1)

        d_kl = (marginal * torch.log(marginal / (self.FB_param + 1e-10))).sum(1)

        if reduction == 'sum':
            cond_entropy = cond_entropy.sum(0)
            d_kl = d_kl.sum(0)
            assert not torch.isnan(cond_entropy), cond_entropy
            assert not torch.isnan(d_kl), d_kl
        elif reduction == 'mean':
            cond_entropy = cond_entropy.mean(0)
            d_kl = d_kl.mean(0)
        return d_kl, cond_entropy, marginal

    def get_ce(self,
               probas: torch.tensor,
               valid_pixels: torch.tensor,
               one_hot_gt: torch.tensor,
               reduction: str = 'sum') -> torch.tensor:
        """
        inputs:
            probas : shape [n_tasks, shot, c, h, w]
            one_hot_gt: shape [n_tasks, shot, num_classes, h, w]
            valid_pixels : shape [n_tasks, shot, h, w]

        updates :
             ce : Cross-Entropy between one_hot_gt and probas, shape [n_tasks,]
        """
        ce = - ((valid_pixels.unsqueeze(2) * (one_hot_gt * torch.log(probas + 1e-10))).sum(2))

        ce = ce.sum(dim=(1, 2, 3))  # [n_tasks]
        ce /= valid_pixels.sum(dim=(1, 2, 3))

        if reduction == 'sum':
            ce = ce.sum(0)
        elif reduction == 'mean':
            ce = ce.mean(0)
        return ce

    def _extract_temporal_protos(self, seq_names: np.ndarray):
        """
        Computes average prototype for all frames in sequence
        Args:
            seq_names: numpy array with sequence names
        returns:
            avg_protos: torch.tensor [n_task x C]
        """
        # avg_prototype: [n_task x n_task x C]
        avg_prototype = self.prototype.unsqueeze(1).repeat(1, seq_names.shape[0], 1)

        # assignment: [n_task x n_task] 0/1 flag to indicate which protos to avg per frame
        assignment = torch.tensor(\
            np.expand_dims(seq_names, axis=0) == np.expand_dims(seq_names, axis=1)).int().cuda()

        avg_prototype = torch.div(avg_prototype * assignment.unsqueeze(2),
                                  assignment.sum(dim=0).unsqueeze(1).unsqueeze(2) + 1e-10)

        avg_prototype = avg_prototype.sum(dim=0)
        return avg_prototype

    def temporal_repri(self, probas, valid_pixels, seq_names):
        loss = torch.zeros(probas.shape[0]).cuda()

        marginal = (valid_pixels.unsqueeze(2) * probas).sum(dim=(1, 3, 4))
        marginal /= valid_pixels.sum(dim=(1, 2, 3)).unsqueeze(1)
        window = self.temporal_window
        if marginal.shape[0] < window:
            return loss

        seq_names_ = np.stack([seq_names[:-2], seq_names[1:-1], seq_names[2:]])
        seq_flags = torch.tensor(
                (seq_names_[0] == seq_names_[1]) * (seq_names_[0] == seq_names_[2])
        ).long().cuda()

        marginal_window = marginal.unfold(0, window, self.temporal_step)
        margin = 0.0
        loss[:marginal_window.shape[0]] = seq_flags * F.relu(torch.abs(\
                marginal_window.unsqueeze(3) - marginal_window.unsqueeze(2)) -margin).sum(1).mean(dim=[1,2])
        return loss

    def get_temporal_loss(self,
                          features: torch.tensor,
                          valid_pixels: torch.tensor,
                          probas: torch.tensor,
                          seq_names: List[str],
                          steering_features: torch.tensor,
                          steering_val_pixels: torch.tensor,
                          steering_mask: torch.tensor,
                          reduction: str = 'none'):
        """
        Compute temporal consistency loss as a regularizer
        Args:
            features: query features [n_task x shot x C x H x W]
            valid_pixels: flag [n_task x shot x H x W]
            probas: predictions on query image [n_task x shot x n_cls x H x W]
            steering_features: support set features [n_task x shot x C x H x W]
            steering_val_pixels: flag [n_task x shot x H x W]
            steering_mask: mask [n_task x shot x n_cls x H x W]
            seq_names: List of sequence names [n_task]
        """
        if type(self.tloss_type) != list:
            self.tloss_type = [self.tloss_type]

        seq_names = np.array(seq_names)
        total_loss = torch.zeros(features.shape[0]).cuda()
        for loss_ in self.tloss_type:
            if loss_ == 'pos_neg_steer_avg':
                # 1- Extract Query Signatures MAP Features with Prob maps
                protos = compute_map(probas, valid_pixels, features)

                # 2- Extract Global Prototype
                avg_protos = self._extract_temporal_protos(seq_names)

                # 3- Compute Global Loss
                pos_loss = temporal_positive_steered(protos, seq_names, avg_protos)
                neg_loss = temporal_negative_steered_spatial(features, probas, valid_pixels, avg_protos)
                loss = pos_loss + neg_loss
            elif loss_ == 'temporal_repri':
                # Compute Local Loss
                loss = self.temporal_repri(probas, valid_pixels, seq_names)

            total_loss += loss

        if reduction == 'mean':
            total_loss = total_loss.mean()

        return total_loss

    def ftune_selected_keyframe(self, all_probas: torch.tensor, all_f_q: torch.tensor, all_f_s: torch.tensor,
            all_gt_s: torch.tensor, all_gt_q: torch.tensor, seqs: List[str], refine_oracle: bool = False):
        """
        Finetune Based on KeyFrames in the sequence
        """
        seqs = np.array(seqs)
        for seq in np.unique(seqs):
            gt_q = all_gt_q[seqs == seq]
            gt_s = all_gt_s[seqs == seq]
            probas = all_probas[seqs == seq]
            f_q = all_f_q[seqs == seq]
            f_s = all_f_s[seqs == seq]

            # TODO: Currently seqs not used Thus this method will only work on YTVIS
            ds_gt_q = F.interpolate(gt_q.float(), size=f_s.size()[-2:], mode='nearest').long()
            ds_gt_s = F.interpolate(gt_s.float(), size=f_s.size()[-2:], mode='nearest').long()
            ds_probas = F.interpolate(probas[:,0], size=f_s.size()[-2:], mode='nearest').unsqueeze(1)

            val_q_pixels = (ds_gt_q != 255).float()
            val_s_pixels = (ds_gt_s != 255).float()

            ds_probas = torch.argmax(ds_probas, dim=2)
            ds_probas_one_hot = to_one_hot(ds_probas, 2)

            protos = compute_map(ds_probas_one_hot, val_q_pixels, f_q)
            one_hot_gt_s = to_one_hot(ds_gt_s, 2)
            if self.keyframe_criteria == 'keyframe_sprt':
                ref_protos = compute_map(one_hot_gt_s, val_s_pixels, f_s)
            elif self.keyframe_criteria == 'keyframe_global':
                ref_protos = self._extract_temporal_protos(seqs)
                ref_protos = ref_protos[np.where(seqs == seq)]

            cossim = F.cosine_similarity(ref_protos, protos, dim=1)
            keyframe_indx = torch.argmax(cossim)

            Nframes = f_q.shape[0]
            ######## For debugging purposes only confirm its learning with real gt of keyframe as upper bound
            val_q_pixels = F.interpolate(val_q_pixels, gt_q.shape[-2:])
            if refine_oracle:
                print('===> Refining with Oracle ================ ')
                pseudogt_keyframe = gt_q[keyframe_indx].unsqueeze(0).repeat(Nframes, 1, 1, 1)
                val_q_pixels = val_q_pixels * (pseudogt_keyframe != 255).float()
                pseudogt_keyframe = to_one_hot(pseudogt_keyframe, 2)
            else:
                pseudogt_keyframe = create_pseudogt(probas[keyframe_indx].unsqueeze(0))
                val_q_pixels = val_q_pixels * (pseudogt_keyframe != 255).float()

                pseudogt_keyframe = to_one_hot(pseudogt_keyframe, 2)
                pseudogt_keyframe = pseudogt_keyframe.repeat(Nframes, 1, 1, 1, 1)
            keyframe_f_q = f_q[keyframe_indx].unsqueeze(0).repeat(Nframes, 1, 1, 1, 1)

            optimizer = torch.optim.SGD([self.prototype, self.bias], lr=self.lr/10.0)

            for iteration in range(1, self.refine_iter):
                logits_q = self.get_logits(keyframe_f_q, seqs=seqs, selected_seq=seq)  # [n_tasks, 1, num_class, h, w]
                keyframe_proba_q = self.get_probas(logits_q, seqs=seqs, selected_seq=seq)

                keyframe_proba_q = F.interpolate(keyframe_proba_q[:, 0], pseudogt_keyframe.shape[-2:])
                keyframe_proba_q = keyframe_proba_q.unsqueeze(1)

                # Ignoring the Padding from Resize (using valid_pixels_q) only, Groundtruth is from preds
                loss = self.get_ce(keyframe_proba_q, val_q_pixels, pseudogt_keyframe, reduction='none')

                optimizer.zero_grad()
                loss.sum(0).backward()
                optimizer.step()

    def TTI(self,
              features_s: torch.tensor,
              features_q: torch.tensor,
              gt_s: torch.tensor,
              gt_q: torch.tensor,
              subcls: List,
              n_shots: torch.tensor,
              seqs: List[str],
              callback: VisdomLogger,
              weights: List[int] = None) -> torch.tensor:
        """
        Performs TTI + RePRI inference

        inputs:
            features_s : shape [n_tasks, shot, c, h, w]
            features_q : shape [n_tasks, shot, c, h, w]
            gt_s : shape [n_tasks, shot, h, w]
            gt_q : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task
            seqs: List of sequence names
            n_shots : # of support shots for each task, shape [n_tasks,]
        updates :
            prototypes : torch.Tensor of shape [n_tasks, num_class, c]

        returns :
            deltas : Relative error on FB estimation right after first update, for each task,
                     shape [n_tasks,]
        """
        deltas = torch.zeros_like(n_shots)
        if weights is None:
            l1, l2, l3, l4 = self.weights
        else:
            l1, l2, l3, l4 = weights

        if l2 == 'auto':
            l2 = 1 / n_shots
        else:
            l2 = l2 * torch.ones_like(n_shots)
        if l3 == 'auto':
            l3 = 1 / n_shots
        else:
            l3 = l3 * torch.ones_like(n_shots)
        original_l4 = l4
        if l4 == 'auto':
            l4 = 1 / n_shots

        self.prototype.requires_grad_()
        self.bias.requires_grad_()
        optimizer = torch.optim.SGD([self.prototype, self.bias], lr=self.lr)

        ds_gt_q = F.interpolate(gt_q.float(), size=features_s.size()[-2:], mode='nearest').long()
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.size()[-2:], mode='nearest').long()

        valid_pixels_q = (ds_gt_q != 255).float()  # [n_tasks, shot, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()  # [n_tasks, shot, h, w]

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [n_tasks, shot, num_classes, h, w]

        for iteration in range(1, self.adapt_iter):
            logits_s = self.get_logits(features_s, seqs=seqs)  # [n_tasks, shot, num_class, h, w]
            logits_q = self.get_logits(features_q, seqs=seqs)  # [n_tasks, 1, num_class, h, w]
            proba_q = self.get_probas(logits_q)
            proba_s = self.get_probas(logits_s)

            d_kl, cond_entropy, marginal = self.get_entropies(valid_pixels_q,
                                                              proba_q,
                                                              reduction='none')
            if original_l4 != 0 and self.enable_temporal:
                tloss = self.get_temporal_loss(features_q, valid_pixels_q, proba_q,
                                               seqs, features_s, valid_pixels_s, one_hot_gt_s)
            else:
                tloss = 0

            ce = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s, reduction='none')
            loss = l1 * ce + l2 * d_kl + l3 * cond_entropy + l4 * tloss

            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()

            # Update FB_param
            if (iteration + 1) in self.FB_param_update  \
                    and ('oracle' not in self.FB_param_type) and (l2.sum().item() != 0):
                deltas = self.compute_FB_param(features_q, gt_q, seqs=seqs).cpu()
                l2 += 1
                self.enable_temporal = True

            if callback is not None and (iteration + 1) % self.visdom_freq == 0:
                self.update_callback(callback, iteration, features_s, features_q, subcls, gt_s, gt_q,
                                     seqs=seqs, tti_flag=(original_l4!=0 and self.enable_temporal))

        return deltas

    def get_mIoU(self,
                 probas: torch.tensor,
                 gt: torch.tensor,
                 subcls: torch.tensor,
                 reduction: str = 'mean') -> torch.tensor:
        """
        Computes the mIoU over the current batch of tasks being processed

        inputs:
            probas : shape [n_tasks, shot, num_class, h, w]
            gt : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task


        returns :
            class_IoU : Classwise IoU (or mean of it), shape
        """
        intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt, self.num_classes)  # [num_tasks, shot, num_class]
        inter_count = defaultdict(int)
        union_count = defaultdict(int)

        for i, classes_ in enumerate(subcls):
            inter_count[0] += intersection[i, 0, 0]
            union_count[0] += union[i, 0, 0]

            if type(classes_) == list:
                for j, class_ in enumerate(classes_):
                    inter_count[class_] += intersection[i, 0, j + 1]  # Do not count background
                    union_count[class_] += union[i, 0, j + 1]
            else:
                inter_count[classes_] += intersection[i, 0, 1]
                union_count[classes_] += union[i, 0, 1]

        class_IoU = torch.tensor([inter_count[subcls] / union_count[subcls] for subcls in inter_count if subcls != 0])
        if reduction == 'mean':
            return class_IoU.mean()
        elif reduction == 'none':
            return class_IoU

    def update_callback(self, callback, iteration: int, features_s: torch.tensor,
                        features_q: torch.tensor, subcls: List[int],
                        gt_s: torch.tensor, gt_q: torch.tensor,
                        seqs: List[str] = None, tti_flag:bool = False) -> None:
        """
        Updates the visdom callback in case live visualization of metrics is desired

        inputs:
            iteration: Current inference iteration
            features_s : shape [n_tasks, shot, c, h, w]
            features_q : shape [n_tasks, shot, c, h, w]
            gt_s : shape [n_tasks, shot, h, w]
            gt_q : shape [n_tasks, shot, h, w]
            subcls : List of classes present in each task
            seqs: List of string sequence names
            tti_flag: Boolean to indicate the use of temporal transductive inference
        returns :
            callback : Visdom logger
        """
        logits_q = self.get_logits(features_q, seqs=seqs)  # [n_tasks, shot, num_class, h, w]
        logits_s = self.get_logits(features_s, seqs=seqs)  # [n_tasks, shot, num_class, h, w]
        proba_q = self.get_probas(logits_q).detach()  # [n_tasks, shot, num_class, h, w]
        proba_s = self.get_probas(logits_s).detach()  # [n_tasks, shot, num_class, h, w]

        f_resolution = features_s.size()[-2:]
        ds_gt_q = F.interpolate(gt_q.float(), size=f_resolution, mode='nearest').long()
        ds_gt_s = F.interpolate(gt_s.float(), size=f_resolution, mode='nearest').long()

        valid_pixels_q = (ds_gt_q != 255).float()  # [n_tasks, shot, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()  # [n_tasks, shot, h, w]

        one_hot_gt_q = to_one_hot(ds_gt_q, self.num_classes)  # [n_tasks, shot, num_classes, h, w]
        oracle_FB_param = (valid_pixels_q.unsqueeze(2) * one_hot_gt_q).sum(dim=(1, 3, 4))
        oracle_FB_param /= (valid_pixels_q.unsqueeze(2)).sum(dim=(1, 3, 4))

        one_hot_gt_s = to_one_hot(ds_gt_s, self.num_classes)  # [n_tasks, shot, num_classes, h, w]


        ce_s = self.get_ce(proba_s, valid_pixels_s, one_hot_gt_s)
        ce_q = self.get_ce(proba_q, valid_pixels_q, one_hot_gt_q)

        mIoU_q = self.get_mIoU(proba_q, gt_q, subcls)

        callback.scalar('mIoU_q', iteration, mIoU_q, title='mIoU')
        if iteration > 0:
            d_kl, cond_entropy, marginal = self.get_entropies(valid_pixels_q,
                                                              proba_q,
                                                              reduction='mean')

            if tti_flag:
                # TODO: Get Consistency type to be passed properly
                tloss = self.get_temporal_loss(features_q, valid_pixels_q, proba_q,
                                               seqs, features_s, valid_pixels_s, one_hot_gt_s,
                                               reduction='mean')
                callback.scalars(['tti'], iteration, [tloss.mean()], title='Temporal Transductive Inference')
            marginal2oracle = (oracle_FB_param * torch.log(oracle_FB_param / marginal + 1e-10)).sum(1).mean()
            FB_param2oracle = (oracle_FB_param * torch.log(oracle_FB_param / self.FB_param + 1e-10)).sum(1).mean()
            callback.scalars(['Cond', 'marginal2oracle', 'FB_param2oracle'], iteration,
                             [cond_entropy, marginal2oracle, FB_param2oracle], title='Entropy')
        callback.scalars(['ce_s', 'ce_q'], iteration, [ce_s, ce_q], title='CE')
