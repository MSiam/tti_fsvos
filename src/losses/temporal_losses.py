import numpy as np
import torch
import torch.nn.functional as F

def temporal_negative_steered_spatial(features, probas, valid_pixels, steering_protos):
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
    """
    # Assignment matrix which frame belongs to which seq 0/1 [n_task x n_task]
    assignment = torch.tensor(\
            np.expand_dims(seq_names, axis=0) == np.expand_dims(seq_names, axis=1)).int().cuda()

    # distance matrix [n_task x n_task]
    distance_matrix = 1 - F.cosine_similarity(protos.unsqueeze(1), steering_protos.unsqueeze(0), dim=2)

    loss = (distance_matrix * assignment).mean(dim=0)
    return loss

def temporal_positive(protos, seq_names):
    assignment = torch.tensor(\
            np.expand_dims(seq_names, axis=0) == np.expand_dims(seq_names, axis=1)).int().cuda()
    assignment = assignment - torch.diag(torch.ones(assignment.shape[0])).cuda()

    # distance matrix
    distance_matrix = 1 - F.cosine_similarity(protos.unsqueeze(1), protos.unsqueeze(0), dim=2)

    # Contrastive loss
    loss = (distance_matrix * assignment).sum(dim=1)
    return loss

def temporal_contrastive(protos, seq_names, nviews, supconloss):
    seqs = np.unique(seq_names)
    seq_ids = {seq: i for i, seq in enumerate(seqs)}
    seq_labels = torch.tensor([seq_ids[seq] for seq in seq_names]).long()

    # Compute Nviews from within frame window
    protos_views = torch.zeros(protos.shape[0], nviews, protos.shape[1]).float().cuda()
    assert nviews == 2, "Not Implemented Yet"
    protos_views[:, 0, :] = protos
    other_view = [i+1 if seq_names[(i+1)%len(seq_names)] == seq else i-1 for i, seq in enumerate(seq_names)]
    protos_views[:, 1, :] = protos[other_view]

    loss = supconloss(protos_views, seq_labels)
    return loss
