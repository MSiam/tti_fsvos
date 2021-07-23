import argparse
import json
import os
import matplotlib.pyplot as plt

from tao import Tao

def combine(list1, list2):
    final_list = list1
    for j in range(len(list2)):
        exist = False
        for i in range(len(list1)):
            if list1[i]['name'] == list2[j]['name']:
                exist = True
        if not exist:
            final_list.append(list2[j])
    return final_list

def merge_tao_annotations(merged_file, main_dir):
    """
    Merge all annotations from training and validation jsons
    Save Merged train_val.json
    """
    json_files = [os.path.join(main_dir, 'annotations/train.json'),
                  os.path.join(main_dir, 'annotations/validation.json')
                  ]
    annotations = {}
    cats_by_split = {}
    for json_file in json_files:
        opened_f = open(json_file)
        annotations_dict = json.load(opened_f)
        for key, _ in annotations_dict.items():
            if key not in annotations:
                annotations[key] = []

            if key != "categories":
                annotations[key] += annotations_dict[key]
            else:
                split = json_file.split('/')[-1].split('.')[0]
                cats_by_split[split] = annotations_dict[key]

    # Combine categories from train and val
    annotations["categories"] = combine(cats_by_split["train"], cats_by_split["validation"])

    json_object = json.dumps(annotations["categories"], indent=4)
    with open("tao_classes.json", 'w') as f:
        f.write(json_object)
    # Dump merged annotations
    json_object = json.dumps(annotations, indent=4)
    with open(merged_file, 'w') as merged_f:
        merged_f.write(json_object)

def load_class_mapping(class_mapping_file):
    with open(class_mapping_file, "r") as f:
        class_mapping = json.load(f)
    return class_mapping

def create_dict_map(tao_dataset, exhaustive_vid_ids, classes):
    """
    Create Videos by Category Dictionary for Sampling Sprt/Qry
    """
    cats_by_vid = {}
    vids_by_cat = {}

    annot_ids = tao_dataset.get_ann_ids(vid_ids=exhaustive_vid_ids)
    annots = tao_dataset.load_anns(ids=annot_ids)

    # vids_by_cat: Create Dictioinary Key: Category Name, Value: Array of Vid Ids
    # cats_by_vid: Create Dictioinary Key: Video ID, Value: Array of Categoriy Ids
    for annot in annots:
        cat_name = tao_dataset.get_name_from_id(annot["category_id"])
        if cat_name not in classes:
            continue

        video_info = tao_dataset.vids[annot["video_id"]]
        if annot["video_id"] not in cats_by_vid:
            cats_by_vid[annot["video_id"]] = set()
        cats_by_vid[annot["video_id"]].add(annot["category_id"])

        if cat_name not in vids_by_cat:
            vids_by_cat[cat_name] = []
        vids_by_cat[cat_name].append(annot["video_id"])

    return cats_by_vid, vids_by_cat

def plot_stats(vids_by_cat, dataset_name):
    count = []
    ticks_labels = []
    for key, value in vids_by_cat.items():
        count.append(len(value))
        ticks_labels.append(key)

    fig, ax = plt.subplots()
    plt.bar(range(len(count)), count, width = 0.5, align='center', color='blue')
    plt.title("TAO-VOS 5i Statistics")
    plt.yscale("log")
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('# Sequences', fontsize=12)

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    N = len(count)
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.xticks(range(len(count)), ticks_labels, rotation=30)
    plt.savefig("%s_stats.png"%dataset_name)

def dump_file_list(vids_by_cat, tao_dataset, out_dir):
    out_file = open(os.path.join(out_dir, 'seqs.txt'), 'w')
    seqs = []
    for key, value in vids_by_cat.items():
        vid_infos = tao_dataset.load_vids(value)

        for vid_info in vid_infos:
            if vid_info['name'] not in seqs:
                seqs.append(vid_info['name'])
                out_file.write(vid_info['name']+'\n')
    out_file.close()

def load_filenames(root, classes, class_mapping_file):
    # Merge Annotations for TAO from Train and Val
    merged_file = os.path.join(root, 'annotations/train_val.json')
    if not os.path.exists(merged_file):
        merge_tao_annotations(merged_file, root)

    tao_dataset = Tao(merged_file)

    # Select Ehaustively Labelled Videos
    exhaustive_vid_ids = []
    for _, video in tao_dataset.vids.items():
        if len(video["not_exhaustive_category_ids"]) == 0:
            exhaustive_vid_ids.append(video["id"])

    tao_dataset.class_mapping = load_class_mapping(class_mapping_file)

    cats_by_vid, vids_by_cat = create_dict_map(tao_dataset, exhaustive_vid_ids, classes)
    dataset_name = class_mapping_file.split('/')[-2]

    out_dir = os.path.join('../../lists/taovos/', dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump_file_list(vids_by_cat, tao_dataset, out_dir)

    plot_stats(vids_by_cat, dataset_name)
    print("# Sequences ", len(cats_by_vid), " For classes ", len(vids_by_cat))

if __name__ == "__main__":
    # PASCAL 16 classes
    # COCO 48 classes

    parser = argparse.ArgumentParser("Tao Tools")
    parser.add_argument("--root", type=str, default="/local/riemann/home/msiam/TAO/")
    parser.add_argument("--class_list", type=str, default="../../lists/pascal/classes.json")
    parser.add_argument("--class_mapping", type=str, default="../../lists/pascal/class_mapping.json")
    args = parser.parse_args()

    with open(args.class_list, 'r') as f:
        classes = json.load(f)
        load_filenames(args.root, classes, args.class_mapping)
