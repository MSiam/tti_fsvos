import numpy as np
import os
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    main_dir = 'dumped_marginals/'
    colors = {'repri': 'b', 'tti': 'r'}

    window = 3
    fig = plt.figure()
    for method in os.listdir(main_dir):
        hist = []
        final_hist = {}
        final_n = {}

        if not os.path.exists(os.path.join('temporal_repri_plots', method)):
            os.makedirs(os.path.join('temporal_repri_plots', method))

        in_dir = os.path.join(main_dir, method)
        for file_ in sorted(os.listdir(in_dir)):
            data = np.load(os.path.join(in_dir, file_), allow_pickle=True).item()
            marginal = data['marginal'].cpu()
            ious = data['miou'].cpu()

            if marginal.shape[0] < 3:
                continue
#            marginal_window = marginal.unfold(0, window, 1)
#            marginal_window = torch.abs(marginal_window.unsqueeze(3) - \
#                                        marginal_window.unsqueeze(2)).sum(1).mean(dim=[1,2])

#            for i in range(marginal_window.shape[0]):
            for i, _ in enumerate(marginal):
                if i == 0:
                    diff = marginal[i+1, 1] - marginal[i, 1]
                elif i == marginal.shape[0] - 1:
                    diff = marginal[i, 1] - marginal[i-1, 1]
                else:
                    diff = marginal[i+1, 1] - marginal[i, 1] / 2.0

                hist.append((np.abs(diff), ious[i]))
#                hist.append((marginal_window[i], ious[i]))
        mindiff = np.min([d[0] for d in hist])
        maxdiff = np.max([d[0] for d in hist])
        for r in np.arange(0, 1, 0.1):
            start = r * (maxdiff-mindiff) + mindiff
            end = (r+0.1) * (maxdiff-mindiff) + mindiff
            if start not in final_hist:
                final_hist[start] = 0
                final_n[start] = 0

            for h in hist:
                if h[0] < end and h[0] > start:
                    final_hist[start] += h[1]
                    final_n[start] += 1

        final_hist = {k: v / final_n[k] for k, v in final_hist.items()}

        plt.plot(final_hist.keys(), final_hist.values(), marker='o', color = colors[method], label=method)

    plt.title('Temporal Fg/Bg Region Proportion Summary')
    plt.xlabel('Difference of ReP')
    plt.ylabel('mIoU')
    plt.legend()

    fig.savefig(os.path.join('temporal_repri_plots', 'summary.png'))

