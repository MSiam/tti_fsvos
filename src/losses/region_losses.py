"""
Taken from https://github.com/Epiphqny/VisTR
which was released under the Apache 2.0 license.
And modified as needed.
"""
"""
VisTR criterion classes.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from bdb import Breakpoint
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def dice_multiclass_loss(inputs, targets, num_classes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.softmax(1).permute(0,2,1)
    onehot_gt = F.one_hot(targets.long(), num_classes=num_classes)

    # Ignore Background Class
    inputs = inputs[:, 1:]
    onehot_gt = onehot_gt[:, 1:]

    numerator = 2 * (inputs * onehot_gt).sum(1)
    denominator = inputs.sum(1) + onehot_gt.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.mean()

def crossentropy_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # Re-organize into the right shape
    prob = inputs
    targets = targets.long()
    C = prob.shape[1]
    prob = prob.permute(0, *range(2, prob.ndim), 1).reshape(-1, C)
    targets = targets.view(-1)

    # Calc the general ce loss
    log_p = F.log_softmax(prob, dim=-1)
    ce = F.nll_loss(log_p, targets, reduction='none')
    # ce = F.cross_entropy(prob, targets, ignore_index=-100)

    # Find the log-prob in the output, regarding the ground-truth label
    all_rows = torch.arange(len(prob))
    log_pt = log_p[all_rows, targets.view(-1)] # why not use [:, targets.view(-1)]?

    # Revert the probability from log-prob
    pt = log_pt.exp()
    focal_term = (1-pt)**gamma

    loss = focal_term * ce

    return loss.mean()

class DiceFocalMultiLabelCriterion(nn.Module):
    """ This class computes the loss for our model.
    The code is based on the code from VisTR.
    """

    def __init__(self, num_classes):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes

    def loss_masks(self, src_masks, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        target_masks = targets.squeeze()
        target_masks = target_masks.to(src_masks)

        # upsample predictions to the target size
        target_size = target_masks.shape[-2:]
        src_masks = F.interpolate(src_masks, size=target_size, mode="bilinear", align_corners=False)

        src_masks = src_masks.flatten(2)
        target_masks = target_masks.flatten(1)

        focal_loss_ = crossentropy_focal_loss(src_masks, target_masks)
        dice_loss_ = dice_multiclass_loss(src_masks, target_masks, self.num_classes)

        losses = {
            "loss_mask": focal_loss_,
            "loss_dice": dice_loss_,
        }
        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        losses.update(self.loss_masks(outputs, targets))
        return losses

