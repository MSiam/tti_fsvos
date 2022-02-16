# Temporal Transductive Inference for Few-shot Video Semantic Segmentation

<div align="center">
<img src="" width="70%" height="70%"><br><br>
</div>

## Getting Started

### Minimum requirements

1. Software :
+ torch==1.9.0
+ numpy==1.18.4
+ cv2==4.2.0
+ pyyaml==5.3.1

For both training and testing, metrics monitoring is done through **visdom_logger** (https://github.com/luizgh/visdom_logger). To install this package with pip, use the following command:

 ```
pip install git+https://github.com/luizgh/visdom_logger.git
 ```
* I modified visdom_logger repo so its best you keep vis_port: -1 in the config files to disable it. If needed let me know and I will send you visdom_logger version I modified which allows passing on env_name and other args to visdom

### Download data

#### Download MiniVSPW used in Pascal-to-MiniVSPW Data

* Download Data from webpage
* expected folder structure

```
VSPW/data
├── seq1
│   ├── origin
│   ├── mask
│   ├── flow
└── seq2
|    ├── origin
|    └── mask ....
```

* Or you can download processed data [here](https://www.dropbox.com/s/a4gqqu0w4t834p2/MiniVSPW.zip?dl=0)

* Instructions from RePRI for downloading processed Pascal Data.
We provide the versions of Pascal-VOC 2012 and MS-COCO 2017 used in this work at https://drive.google.com/file/d/1Lj-oBzBNUsAqA9y65BDrSQxirV8S15Rk/view?usp=sharing. You can download the full .zip and directly extract it at the root of this repo.

#### Download Full VSPW for MiniVSPW-to-MiniVSPW evaluation
* use processed data from [VSPW_480](https://github.com/sssdddwww2/vspw_dataset_download)

##### Download YTVIS dataset
* use 2019 YTVIS version similar to [DANet](https://github.com/scutpaul/DANet)

#### About the train/val splits

The train/val splits are directly provided in lists/. How they were obtained is explained at https://github.com/Jia-Research-Lab/PFENet

### Download pre-trained models

#### Pre-trained backbones
First, you will need to download the ImageNet pre-trained backbones from RePRI at [here](https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v) and put them under initmodel/. These will be used if you decide to train your models from scratch.

#### Pre-trained models
* For Pascal-to-MiniVSPW : use RePRI ones "directly provide the [full pre-trained models](https://drive.google.com/file/d/1iuMAo5cJ27oBdyDkUI0JyGIEH60Ln2zm/view?usp=sharing). You can download them and directly extract them at the root of this repo. This includes Resnet50 and Resnet101 backbones on Pascal-5i, and Resnet50 on Coco-20i."

* For YTVIS: use these provided [models](https://www.dropbox.com/s/2q6vqnrkpjju0yc/model_ckpt_ytvis.zip?dl=0)
* For YTVIS (with auxiliary DCL): use these provided [models](https://www.dropbox.com/s/qynq5lot2696ogm/model_ckpt_dcl.zip?dl=0)
* For MiniVSPW-to-MiniVSPW: use these provided [models](https://www.dropbox.com/s/ymgzphqo4b5fddb/checkpoints_nminivspw.zip?dl=0)

## Overview of the repo

Data are located in data/. All the code is provided in src/. Default configuration files can be found in config_files/. Training and testing scripts are located in scripts/. Lists/ contains the train/validation splits for each dataset.

## Inference
* Reproduce results for all tables
```
bash scripts/test_all_datasets.sh
```
## Training

If you want to use the pre-trained models, this step is optional. Otherwise, you can train your own models from scratch with the scripts/train.sh script, as follows.

```python
bash scripts/train.sh {data} {fold} {[gpu_ids]} {layers}
```
For instance, if you want to train a Resnet50-based model on the fold-0 of Pascal-5i on GPU 1, use:
```python
bash scripts/train.sh pascal 0 [1] 50
```

For training on ytvis standard training
```python
bash scripts/train.sh ytvis 0 [1] 50
```

Note that this code supports distributed training. If you want to train on multiple GPUs, you may simply replace [1] in the previous examples with the list of gpus_id you want to use.


## Acknowledgments

We gratefully thank the authors of https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation for building upon their code.
We also rely on https://github.com/scutpaul/DANet, for understanding their Youtube-VIS episodic version.
