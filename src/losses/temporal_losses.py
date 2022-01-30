import numpy as np
import torch
import torch.nn.functional as F

def correlation_based_consistency(f, p, valid, roi_grids, h, w):
#    m1 = p1.argmax(dim=0)
#    m2 = p2.argmax(dim=0)
    """
    Compute Shiffted Correlation Method
    f: features [Nw x N_te x C x H x W]
    p: probabilities [Nw x N_te x 2 x H x W]
    valid: valid pixels [Nw x NN_te x H x W]
    roi_grids: List[tensor] [HWxHW]
    h, w : feature size
    """
    assert f.shape[0] == 2, "Ony supporting temporal window 2"

    f1 = f[0].view((*f.shape[1:3],1, -1))
    f2 = f[1].view((*f.shape[1:3], -1, 1))

    p1 = p[0]
    p2 = p[1]

    valid = valid[0]
    minibatch = 5
    loss = torch.tensor([0.0]).cuda()
    # FOr the sake of memory compute loss on minibatch size B
    for i in range(0, f1.shape[0], minibatch):
        end = minibatch
        if i + end >= f1.shape[0]:
            end = f1.shape[0] - i

        # B x HW x HW correlation map
        correlation = F.cosine_similarity(f1[i:i+end], f2[i:i+end], dim=1)
        # Compute for every shifted windows (9 windows) maxcorr and peaks
        max_corrs = []
        final_peaks = []
        for j, roi_grid in enumerate(roi_grids):
            grid_correlation = correlation * roi_grid.unsqueeze(0)
            max_corr, peaks = torch.max(grid_correlation, dim=1)
            max_corrs.append(max_corr)
            final_peaks.append(peaks)
        # Compute max corr and peak from these peaks and variances
        max_corrs = torch.stack(max_corrs)
        indices = torch.argmax(max_corrs, dim=0)
        final_peaks = torch.stack(final_peaks)
        mean = final_peaks.float().mean(dim=0)
        variances = 1.0/( ((final_peaks - mean)**2).mean(dim=0) + 1e-10)
        # Index x,y peaks
        restx, resty = torch.meshgrid(torch.arange(final_peaks.shape[1]),
                                      torch.arange(final_peaks.shape[2]))
        peaks = final_peaks[indices, restx, resty]
        xpeaks = (peaks % w).view(peaks.shape[0], h,w)
        ypeaks = torch.div(peaks, w, rounding_mode='floor').view(peaks.shape[0], h,w)
        # Sum over classes dimension (probabilities..) and avg on pixels
        # loss_perpixel = (p2[:,ypeaks, xpeaks] * \
        # torch.log(p2[:,ypeaks, xpeaks] / (p1 + 1e-10))).sum(0)
        restb, _, _ = torch.meshgrid(torch.arange(i, i+end), torch.arange(p2.shape[2]), torch.arange(p2.shape[3]))
        p2_fg = p2[:,1]; p1_fg = p1[:, 1]
        p2_bg = p2[:,0]; p1_bg = p1[:, 0]

        # Perpixel loss reweighted with variances of matching
        loss_perpixel = torch.abs((p2_fg[restb, ypeaks, xpeaks] - p1_fg[i:i+end])) + \
                        torch.abs((p2_bg[restb, ypeaks, xpeaks] - p1_bg[i:i+end]))
        loss += (loss_perpixel * variances.reshape(minibatch, h, w) * valid[i:i+end]).sum() / \
                    ((variances.reshape(minibatch, h, w) * valid[i:i+end]).sum() + 1e-10)
    return loss

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
