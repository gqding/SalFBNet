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

### 3. Datasets

Dataset | #Image | #Training | #Val. | #Testing | Size | URL | Paper
:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
SALICON | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [download link](http://salicon.net/challenge-2017/) | [paper](http://www-users.cs.umn.edu/~qzhao/publications/pdf/salicon_cvpr15.pdf)
MIT300 | 300 | - | - | 300 | ~4GB | [download link](http://saliency.mit.edu/results_mit300.html) | [paper](https://dspace.mit.edu/handle/1721.1/68590)  
MIT1003 | 1003 | 900* | 103 | - | ~4GB | [link](http://people.csail.mit.edu/tjudd/WherePeopleLook/) | [paper](http://people.csail.mit.edu/tjudd/WherePeopleLook/Docs/wherepeoplelook.pdf)
PASCAL-S | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [download link](http://cbs.ic.gatech.edu/salobj/) | [paper](https://arxiv.org/pdf/1406.2807)
DUT-OMRON | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [download link](http://saliencydetection.net/dut-omron/) | [paper](http://saliencydetection.net/dut-omron/download/manifold.pdf)
TORONTO | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [download link](http://www-sop.inria.fr/members/Neil.Bruce/) | [paper](http://www-sop.inria.fr/members/Neil.Bruce/NIPS2005_0081.pdf)
Pseudo-Saliency (Ours) | 176,880 | 150,000 | 26,880 | - | ~24.2GB | [download link]() | [paper]()


## Performance Evaluation

<img src="Figs/visualization.png" alt="input" style="width:600px">

<img src="Figs/performance123.png" alt="input" style="width:600px">

<img src="Figs/performance4.png" alt="input" style="width:600px">

<img src="Figs/performance5.png" alt="input" style="width:600px">

<img src="Figs/efficiency.png" alt="input" style="width:600px">
