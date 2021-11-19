# SalFBNet

SalFBNet: Learning Pseudo-Saliency Distribution via Feedback Convolutional Networks, 2021

Guanqun Ding, Nevrez Imamoglu, Ali Caglayan, Masahiro Murakawa, Ryosuke Nakamura

<img src="Figs/salfbnet.png" alt="input" style="width:600px">

## Getting Started
### 1. Installation
```
conda create -n salfbnet python=3.8
conda activate salfbnet
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install scikit-learn scipy tensorboard tqdm
pip install torchSummeryX
```
### 2. Run
Runing codes will be released after our paper pablished.

## Datasets

Dataset | #Image | #Training | #Val. | #Testing | Size | URL | Paper
:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
SALICON | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [link]() | [paper]()
MIT300 | 300 | - | - | 300 | ~4GB | [link]() | [paper]()
MIT1003 | 1003 | 900* | 103 | - | ~4GB | [link]() | [paper]()
PASCAL-S | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [link]() | [paper]()
DUT-OMRON | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [link]() | [paper]()
TORONTO | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [link]() | [paper]()
Pseudo-Saliency (Ours) | 176,880 | 150,000 | 26,880 | - | ~24.2GB | [link]() | [paper]()


## Performance Evaluation

<img src="Figs/visualization.png" alt="input" style="width:600px">

<img src="Figs/performance123.png" alt="input" style="width:600px">

<img src="Figs/performance4.png" alt="input" style="width:600px">

<img src="Figs/performance5.png" alt="input" style="width:600px">

<img src="Figs/efficiency.png" alt="input" style="width:600px">
