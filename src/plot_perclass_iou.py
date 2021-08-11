import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    main_dir = '/home/menna/Code/temporal_fss/results_proto_rep_tti/test/arch=resnet-50/data=ytvis_episodic/shot=shot_5/'

    perclass_results = {}
    for split_dir in sorted(os.listdir(main_dir)):
        start_parsing = False
        with open(os.path.join(main_dir, split_dir, 'log_.txt'), 'r') as f:
            for line in f:
                if "========= Method" in line:
                    method = line.replace("=", "").split("Method")[1].strip()
                    if method not in perclass_results:
                        perclass_results[method] = {}
                    start_parsing = True
                elif "mIoU---" in line:
                    continue
                elif "Test:" in line or "Final" in line:
                    start_parsing = False
                elif start_parsing:
                    cls_name, cls_iou = line.split(':')
                    cls_name = int(cls_name.split(' ')[1])
                    if cls_name not in perclass_results[method]:
                        perclass_results[method][cls_name] = []
                    perclass_results[method][cls_name].append(float(cls_iou))

    plt.title("Per-Class IoU Youtube-VOS")
    colors = {'tti': 'r', 'repri': 'b'}

    for method in perclass_results.keys():
        if method == 'tti':
            w = 0.4
        else:
            w = 0
        perclass_results[method] = {k: np.mean(v) for k, v in perclass_results[method].items()}
        plt.bar(np.array(list(perclass_results[method].keys()))+w, perclass_results[method].values(),
                width=0.4, align = 'edge', color=colors[method], label=method)

    plt.legend()
    plt.xlabel("Classes")
    plt.ylabel("IoU")
    plt.savefig("perclass_plot.png")
