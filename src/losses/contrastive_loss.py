import torch
import torch.nn.functional as F

import torch
import torch.nn as nn


# Code Source: DenseCL, https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/heads/contrastive_head.py
class ContrastiveHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1, rank=0):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.temperature = temperature
        self.rank = rank

    def forward(self, pos, neg, invalid=None):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
            invalid: Nx1
        Returns:
            Tensor: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(self.rank)
        if invalid is not None:
            labels[invalid] = 255
        loss = self.criterion(logits, labels)
        return loss

def dense_temporal_contrastive_loss(features: torch.tensor, temperature: float, rank: int,
                                    aux_gt: torch.tensor):
    """
    Features: [B x TWindow x C x H x W]
    temperature: used in contrastive loss
    rank: CUDA device
    aux_gt: [B x Nseq x TWindow x 1 x H x W]tensor with [0, 255] to highlight invalid pixels
    """

    # Compute Cosine SImilarity along temporal window
    fshape = features.shape
    features = features.view(*fshape[:3], -1)
    cossim = (features[:,0]).permute(0,2,1).matmul(features[:,1])

    # Reshape aux gt as features to retrieve invalid pixels
    aux_gt = aux_gt.view(-1, *aux_gt.shape[-3:])
    aux_gt = F.interpolate(aux_gt.float(), fshape[-2:], mode='nearest')
    aux_gt = aux_gt.view(*fshape[:2], -1)
    invalid = (aux_gt[:, 0] == 255)

    # Maximum Similarity
    dense_pos, pos_indices = torch.max(cossim, dim=2)

    # Compute positives, negatives
    indices_all = torch.arange(features.shape[-1]).view(1, 1, features.shape[-1])
    indices_all = indices_all.expand_as(cossim).to(rank)
    neg_indices = pos_indices.unsqueeze(2).expand_as(indices_all) != indices_all
    dense_neg = cossim[neg_indices].view(*cossim.shape[:2], cossim.shape[-1] - 1)

    # Compute loss
    contrastive_head = ContrastiveHead(temperature=temperature, rank=rank)
    loss = contrastive_head(dense_pos.view(-1, 1), dense_neg.view(-1, cossim.shape[-1]-1),
                            invalid.view(-1))
    return loss
