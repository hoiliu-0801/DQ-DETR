# DQ-DETR: DETR with Dynamic Query for Tiny Object Detection

![method](figure/model_final_V4.jpg)

* This repository is an official implementation of the paper DQ-DETR: DETR with Dynamic Query for Tiny Object Detection.
* The original repository link was https://github.com/Katie0723/DQ-DETR. Here is the updated link.

## News

[2024/12/06]: We released the organized datasets AI-TOD-V1 and AI-TOD-V2.

[2024/7/1]: **DQ-DETR** has been accepted by **ECCV 2024**. 🔥🔥🔥

[2024/5/3]: **DNTR** has been accepted by **TGRS 2024**. 🔥🔥🔥

## Installation -- Compiling CUDA operators

* The code are built upon the official [DINO](https://github.com/IDEA-Research/DINO) repository.

```sh
conda env create -f environment.yml
conda activate dqdetr

# compile CUDA operators
cd models/dqdetr/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

<!-- # bash scripts/DQ_eval.sh /nfs/home/hoiliu/dqdetr/weights/dqdetr_best305.pth -->

## Eval models

```sh
scripts/DQ_eval.sh /path/to/your/dataset /path/to/your/checkpoint
```

## Trained Model

* Changed the pretrained model path in DQ.sh

```sh
CUDA_VISIBLE_DEVICES=5,6,7 bash scripts/DQ.sh /path/to/your/dataset
```

## Our works on Tiny Object Detection

| Title       | Venue     | Links                                                                                       | 
|-------------|-----------|---------------------------------------------------------------------------------------------|
| **DNTR**    | TGRS 2024 | [Paper](https://arxiv.org/abs/2406.05755) \| [code](https://github.com/hoiliu-0801/DNTR)    | 
| **DQ-DETR** | ECCV 2024 | [Paper](https://arxiv.org/abs/2404.03507) \| [code](https://github.com/hoiliu-0801/DQ-DETR) | 

## Performance

Table 1. **Training Set:** AI-TOD-V2 trainval set, **Testing Set:** AI-TOD-V2 test set, 36 epochs, where FRCN, DR
denotes Faster R-CNN and DetectoRS, respectively.

|    Method    | Backbone |   mAP    | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>vt</sub> | AP<sub>t</sub> | AP<sub>s</sub> | AP<sub>m</sub> | 
|:------------:|:--------:|:--------:|:---------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|
| Faster R-CNN |   R-50   |   11.1   |      26.3       |       7.6       |       0.0       |      7.2       |      23.3      |      33.6      | 
|   NWD-RKA    |   R-50   |   23.4   |      53.5       |      16.8       |       8.7       |      23.8      |      28.5      |      36.0      |
|   DAB-DETR   |   R-50   |   22.4   |      55.6       |      14.3       |       9.0       |      21.7      |      28.3      |      38.7      | 
|  DINO-DETR   |   R-50   |   25.9   |      61.3       |      17.5       |      12.7       |      25.3      |      32.0      |      39.7      | 
|   DQ-DETR    |   R-50   | **30.5** |    **69.2**     |    **22.7**     |    **15.2**     |    **30.9**    |    **36.8**    |    **45.5**    | 

## AI-TOD-v1 and AI-TOD-v2 Datasets

* Step 1: Download the datasets from the below link.

```sh
https://drive.google.com/drive/folders/1hkbcZ3TPABx3QxoCufE1KAPu55Ibw-8d?usp=sharing
```

* Step 2: Organize the downloaded files in the following way.

```text
├─ Dataset
│   └─ aitod
│       ├─ annotations
│       ├─ images
│       ├─ test
│       ├─ train
│       ├─ trainval
│       └─ val
├─ DQ-DETR
```

## Pretrained Weights

Download checkpoint from https://reurl.cc/NlvV2Q.

1. pretrain_model.pth ⇒ pretained model
2. dqdetr_best305.pth ⇒ DQ-DETR model on AITOD-V2 with 30.5 AP

## Citation

```bibtex
@InProceedings{huang2024dq,
    author = {Huang, Yi-Xin and Liu, Hou-I and Shuai, Hong-Han and Cheng, Wen-Huang},
    title = {DQ-DETR: DETR with Dynamic Query for Tiny Object Detection},
    booktitle = {European Conference on Computer Vision},
    pages = {290--305},
    year = {2025},
    organization = {Springer}
}

@ARTICLE{10518058,
    author = {Liu, Hou-I and Tseng, Yu-Wen and Chang, Kai-Cheng and Wang, Pin-Jyun and Shuai, Hong-Han and Cheng, Wen-Huang},
    journal = {IEEE Transactions on Geoscience and Remote Sensing},
    title = {A DeNoising FPN With Transformer R-CNN for Tiny Object Detection},
    year = {2024},
    volume = {62},
    number = {},
    pages = {1-15},
}
```