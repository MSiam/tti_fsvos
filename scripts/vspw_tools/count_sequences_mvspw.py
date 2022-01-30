import numpy as np
import os
from os.path import join as osp
from PIL import Image
import json
from tqdm import tqdm

main_dir = "/local/riemann/home/msiam/VSPW_All_Parsed/VSPW_480p/"

################################### Parse Train/Val Seqs###########################
seqs = []
for fname in ['train.txt', 'val.txt']:
    fname = osp(main_dir, fname)
    with open(fname, 'r') as f:
        for line in f:
            seqs.append(line.strip())

print('Total number of sequences in train/val is ', len(seqs))

############################ Parse Classes
with open(osp(main_dir, 'label_num_dic_final.json')) as f:
    classes = json.load(f)

classes_rev = {v: k for k, v in classes.items()}
classes_stuffl = {'wall': 1, 'ceiling': 1, 'door': 1, 'stair': 1, 'ladder': 1, 'escalator': 1, 'Playground_slide': 1, 'handrail_or_fence': 1, 'window': 1, 'others': 1, 'rail': 1, 'goal': 0, 'pillar': 0, 'pole': 0, 'floor': 1, 'ground': 1, 'grass': 1, 'sand': 1, 'athletic_field': 1, 'road': 1, 'path': 1, 'crosswalk': 1, 'building': 1, 'house': 1, 'bridge': 1, 'tower': 1, 'windmill': 1, 'well_or_well_lid': 1, 'other_construction': 1, 'sky': 1, 'mountain': 1, 'stone': 1, 'wood': 1, 'ice': 1, 'snowfield': 1, 'grandstand': 1, 'sea': 1, 'river': 1, 'lake': 1, 'waterfall': 1, 'water': 1, 'billboard_or_Bulletin_Board': 0, 'sculpture': 0, 'pipeline': 1, 'flag': 0, 'parasol_or_umbrella': 0, 'cushion_or_carpet': 0, 'tent': 0, 'roadblock': 1, 'car': 0, 'bus': 0, 'truck': 0, 'bicycle': 0, 'motorcycle': 0, 'wheeled_machine': 1, 'ship_or_boat': 0, 'raft': 0, 'airplane': 0, 'tyre': 0, 'traffic_light': 0, 'lamp': 0, 'person': 0, 'cat': 0, 'dog': 0, 'horse': 0, 'cattle': 0, 'other_animal': 1, 'tree': 1, 'flower': 0, 'other_plant': 1, 'toy': 0, 'ball_net': 0, 'backboard': 0, 'skateboard': 0, 'bat': 0, 'ball': 0, 'cupboard_or_showcase_or_storage_rack': 1, 'box': 0, 'traveling_case_or_trolley_case': 0, 'basket': 0, 'bag_or_package': 0, 'trash_can': 0, 'cage': 0, 'plate': 0, 'tub_or_bowl_or_pot': 0, 'bottle_or_cup': 0, 'barrel': 0, 'fishbowl': 0, 'bed': 1, 'pillow': 0, 'table_or_desk': 0, 'chair_or_seat': 0, 'bench': 0, 'sofa': 0, 'shelf': 1, 'bathtub': 1, 'gun': 0, 'commode': 1, 'roaster': 1, 'other_machine': 1, 'refrigerator': 0, 'washing_machine': 0, 'Microwave_oven': 0, 'fan': 0, 'curtain': 1, 'textiles': 1, 'clothes': 1, 'painting_or_poster': 1, 'mirror': 1, 'flower_pot_or_vase': 0, 'clock': 0, 'book': 0, 'tool': 1, 'blackboard': 0, 'tissue': 0, 'screen_or_television': 0, 'computer': 0, 'printer': 0, 'Mobile_phone': 0, 'keyboard': 0, 'other_electronic_product': 1, 'fruit': 0, 'food': 1, 'instrument': 1, 'train': 0}


class_hist = {int(c): 0 for cname, c in classes.items()}

################################### Parse Masks and Compute Histogram
#for seq in tqdm(seqs):
#    sq_dir = osp(main_dir, 'data', seq, 'mask')
#    for fname in sorted(os.listdir(sq_dir)):
#        mask = np.array(Image.open(osp(sq_dir, fname)))
#        for c in np.unique(mask):
#            if c not in class_hist:
#                if c != 255:
#                    print('not exists ', c)
#                continue
#
#            class_hist[c] += 1
#np.save('vspw_class_hist.npy', class_hist)

################################### Parse Masks and Compute AvgSize
#class_sizes = {int(c): 0 for cname, c in classes.items()}
#class_n = {int(c): 0 for cname, c in classes.items()}
#for seq in tqdm(seqs):
#    sq_dir = osp(main_dir, 'data', seq, 'mask')
#    for fname in sorted(os.listdir(sq_dir)):
#        mask = np.array(Image.open(osp(sq_dir, fname)))
#        for c in np.unique(mask):
#            if c not in class_sizes:
#                if c != 255:
#                    print('not exists ', c)
#                continue
#            class_sizes[c] += len(mask[mask==c])
#            class_n[c] += 1
#class_sizes = {k: v / (class_n[k] + 1e-10) for k, v in class_sizes.items()}
#np.save('vspw_class_sizes.npy', class_sizes)

################################### Pick TopK classes and split into train/val/test in 4 folds
vspw_class_hist = np.load('vspw_class_hist.npy', allow_pickle=True).item()
vspw_class_sizes = np.load('vspw_class_sizes.npy', allow_pickle=True).item()
vspw_class_hist = {k: v for k, v in vspw_class_hist.items() if not classes_stuffl[classes_rev[str(k)]]}
vspw_class_hist = {k: v for k, v in sorted(vspw_class_hist.items(), key=lambda item: item[1], reverse=True)}

min_size = 10000
vspw_class_hist = {k: v for k, v in vspw_class_hist.items() if vspw_class_sizes[k]> min_size}

N = 50
vspw_class_hist = {k:v for i, (k, v) in enumerate(vspw_class_hist.items()) if i < N}
vspw_class_hist = {k:v for k, v in sorted(vspw_class_hist.items(), key=lambda item: item[0], reverse=False)}
vspw_class_names = [classes_rev[str(k)] for k, v in vspw_class_hist.items()]
ntest = 10

################################ Save classes per fold
def dump_classes(out_dir, split, foldno, class_list, classes):
    dic = {int(classes[c]): c for c in class_list}
    class_list_ids = [int(classes[c]) for c in class_list]
    with open(osp(out_dir, 'class_%s_%d.json'%(split, foldno)), 'w') as f:
        json.dump(dic, f)
    return class_list_ids

################################ Save "vidname filename" per fold
def dump_files(out_dir, split, foldno, class_list, imgs_dir):
    seqs = []
#    for fname in ['train.txt', 'val.txt']:
    fname = 'train.txt' if split == 'train' else 'val.txt'
    fname = osp(imgs_dir, fname)
    with open(fname, 'r') as f:
        for line in f:
            seqs.append(line.strip())

    file_list = []
    for seq in tqdm(seqs):
        sq_dir = osp(imgs_dir, 'data', seq, 'mask')
        for fname in sorted(os.listdir(sq_dir)):
            mask = np.array(Image.open(osp(sq_dir, fname)))
            for c in np.unique(mask):
                if c in class_list:
                    if len(mask[mask==c]) > 100: #10x10 square area
                        file_list.append( (seq, osp(seq, 'mask', fname)) )
                        break

    with open(osp(out_dir, '%s_%d.txt'%(split, foldno)), 'w') as f:
        for entry in file_list:
            f.write(entry[0] + ' ' + entry[1] + '\n')

############################## Create all folds files
class_dir = 'lists/nminivspw_large/'
for i in range(4):
    test_classes = vspw_class_names[i*ntest:(i+1)*ntest]
    test_cls_ids = dump_classes(class_dir, 'test', i, test_classes, classes)

    if (i+2)%4*ntest == 0:
        val_classes = vspw_class_names[(i+1)%4*ntest:(i+2)*ntest]
    else:
        val_classes = vspw_class_names[(i+1)%4*ntest:(i+2)%4*ntest]
    val_cls_ids = dump_classes(class_dir, 'val', i, val_classes, classes)

    train_classes = set(vspw_class_names) - set(test_classes + val_classes)
    train_cls_ids = dump_classes(class_dir, 'train', i, train_classes, classes)

    dump_files(class_dir, 'train', i, train_cls_ids, main_dir)
    dump_files(class_dir, 'val', i, val_cls_ids, main_dir)
    dump_files(class_dir, 'test', i, test_cls_ids, main_dir)
