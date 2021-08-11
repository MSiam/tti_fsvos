import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    main_dir = 'dumped_marginals/'

    for method in os.listdir(main_dir):
        if not os.path.exists(os.path.join('temporal_repri_plots', method)):
            os.makedirs(os.path.join('temporal_repri_plots', method))

        in_dir = os.path.join(main_dir, method)
        for file_ in sorted(os.listdir(in_dir)):
            data = np.load(os.path.join(in_dir, file_), allow_pickle=True).item()
            marginal = data['marginal'].cpu()
            ious = data['miou'].cpu()
            fig = plt.figure()

            plt.plot(range(marginal.shape[0]), marginal[:,0], marker='o', color = 'r', label='Marginal Bg')
            plt.plot(range(marginal.shape[0]), marginal[:,1], marker='o', color='b', label='Marginal Fg')
            plt.plot(range(marginal.shape[0]), ious, marker='o', color='c', label='IoU')

            plt.title('Temporal Fg/Bg Region Proportion, Seq: %s'%file_.split('.')[0])
            plt.xlabel('Frame #')
            plt.ylabel('Measurement')
            plt.legend()

            fig.savefig(os.path.join('temporal_repri_plots', method, file_.replace('npy', 'png')))
            plt.clf()

