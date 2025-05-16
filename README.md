# Temporal Transductive Inference for Few-shot Video Semantic Segmentation
Full length paper accepted in IJCV 2025, Short paper accepted in ML4AD Workshop in NeurIPS 2021. 

<div align="center">
<img src="https://github.com/MSiam/tti_fsvos/blob/1bcf3b4de872ce5ab5f18afca57bec6293511627/figures/IJCV_TTI_Overview.png" width="60%" height="100%"><br><br>
</div>

## Getting Started

### Minimum requirements

1. Software :
+ torch==1.9.0


For both training and testing, metrics monitoring is done through **visdom_logger** (https://github.com/luizgh/visdom_logger). To install this package with pip, use the following command:

 ```
pip install git+https://github.com/luizgh/visdom_logger.git
pip install git+https://github.com/youtubevos/cocoapi.git
pip install -r requirements.txt
 ```
* I use a modified version of visdom_logger so its best you keep visdom_port: -1 in the config files to disable it, and use instead wandb for monitoring training. 

### Download data

#### Download MiniVSPW used in Pascal-to-MiniVSPW Data

* Download processed data [here](https://www.dropbox.com/s/a4gqqu0w4t834p2/MiniVSPW.zip?dl=0)
* Expected folder structure:

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

* Use the instructions from RePRI for downloading [processed Pascal Data] (https://drive.google.com/file/d/1Lj-oBzBNUsAqA9y65BDrSQxirV8S15Rk/view?usp=sharing).

#### Download Full VSPW for MiniVSPW-to-MiniVSPW evaluation
* use processed data from [VSPW_480](https://github.com/sssdddwww2/vspw_dataset_download)
* Example explaining why we specificlly pick PASCAL classes and ignore background (i.e. stuff classes). We show the effect of not removing stuff classes (such as road, building, ...) during training and how it would lead to contaminating the learning process with the novel class boundaries (in our case the person class). While removing the stuff classes will ensure that does not occur which explains our choice of classes.

<div align="center">
<img src="https://github.com/MSiam/tti_fsvos/blob/main/figures/MiniVSPW_Example_Leak_Boundaries.png" width="80%" height="60%"><br><br>
</div>


##### Download YTVIS dataset
* Use 2019 YTVIS version similar to [DANet](https://github.com/scutpaul/DANet)

#### About the train/val splits

The train/val splits are directly provided in lists/.

### Download pre-trained models

#### Pre-trained backbones
First, you will need to download the ImageNet pre-trained backbones from RePRI at [here](https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v) and put them under initmodel/. These will be used if you decide to train your models from scratch.

#### Trained models
* For **Pascal-to-MiniVSPW** : use RePRI ones "[full pre-trained models](https://drive.google.com/file/d/1iuMAo5cJ27oBdyDkUI0JyGIEH60Ln2zm/view?usp=sharing)."
* For **YTVIS**: use these provided [models](https://www.dropbox.com/s/2q6vqnrkpjju0yc/model_ckpt_ytvis.zip?dl=0)
* For **YTVIS (with auxiliary DCL)**: use these provided [models](https://www.dropbox.com/s/qynq5lot2696ogm/model_ckpt_dcl.zip?dl=0)
* For **MiniVSPW-to-MiniVSPW**: use these provided [models](https://www.dropbox.com/s/ymgzphqo4b5fddb/checkpoints_nminivspw.zip?dl=0)

## Overview of the repo

All the code is provided in src/. Default configuration files can be found in config_files/. Training and testing scripts are located in scripts/. Lists/ contains the train/validation splits for each dataset.

## Inference
* Reproduce results for all tables
```
bash scripts/test_all_datasets.sh
```
## Training

You can train your own models from scratch with the scripts/train.sh script, as follows.

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

## References
Please city my paper if you find it useful in your research
```
@article{siamijcv2025,
	author = {Siam, Mennatullah},
	date = {2025/03/06},
	date-added = {2025-05-16 14:50:33 +0300},
	date-modified = {2025-05-16 14:50:33 +0300},
	doi = {10.1007/s11263-025-02390-x},
	id = {Siam2025},
	isbn = {1573-1405},
	journal = {International Journal of Computer Vision},
	title = {Temporal Transductive Inference for Few-Shot Video Object Segmentation},
	url = {https://doi.org/10.1007/s11263-025-02390-x},
	year = {2025}}
```
