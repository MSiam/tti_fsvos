import numpy as np
import torch
import torch.nn.functional as F

def temporal_negative_steered_spatial(features, probas, valid_pixels, steering_protos):
    """
    Args:
        features: [B x shot x C x H x W]
        probas: [B x shot x 2 x H x W]
        valid_pixels: [B x shot x H x W]
        steering_protos: [B x C]
    """
    bg_probas = (1 - probas) * valid_pixels.unsqueeze(2)

    # distance matrix
    shape = steering_protos.shape

    distance_matrix = bg_probas.squeeze(1) * (F.cosine_similarity(features, \
                            steering_protos.view(shape[0], 1, shape[1], 1, 1), dim=2))
    distance_matrix = torch.max(torch.zeros(distance_matrix.shape).cuda(), distance_matrix)

    loss = distance_matrix.sum(dim=(1, 2,3)) / (bg_probas.squeeze(1).sum(dim=(1, 2,3)) + 1e-10)
    return loss

def temporal_positive_steered(protos, seq_names, steering_protos):
    """
    Use prototypes extracted from predicted masks on query to be minimized
    to the average prototype learned on all frames in a sequence
    Args:
        protos: [B x C]
        seq_names : [str] * B
        steering_protos: [B x C]
    """
    # Assignment matrix which frame belongs to which seq 0/1 [n_task x n_task]
    assignment = torch.tensor(\
            np.expand_dims(seq_names, axis=0) == np.expand_dims(seq_names, axis=1)).int().cuda()

    # distance matrix [n_task x n_task]
    distance_matrix = 1 - F.cosine_similarity(protos.unsqueeze(1), steering_protos.unsqueeze(0), dim=2)

    loss = (distance_matrix * assignment).mean(dim=0)
    return loss
