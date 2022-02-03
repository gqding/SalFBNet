# SalFBNet

![](https://img.shields.io/static/v1?label=python&message=3.8&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.10.0&color=<COLOR>)
![](https://img.shields.io/static/v1?label=torchvision&message=0.11.1&color=<COLOR>)
![](https://img.shields.io/static/v1?label=torchsummaryX&message=1.3.0&color=<COLOR>)
![](https://img.shields.io/static/v1?label=opencv-python&message=3.4.15.55&color=<COLOR>)
![](https://img.shields.io/static/v1?label=cuda&message=11.3&color=<COLOR>)

This repository includes Pytorch implementation for the following paper:

SalFBNet: Learning Pseudo-Saliency Distribution via Feedback Convolutional Networks, 2021. ([pdf](https://arxiv.org/pdf/2112.03731.pdf))

Guanqun Ding, Nevrez Imamoglu, Ali Caglayan, Masahiro Murakawa, Ryosuke Nakamura

<img src="Figs/salfbnet.png" alt="input" style="width:600px">

### Citation
Please cite the following papers if you use our data or codes in your research.

```
@misc{ding2021salfbnet,
      title={SalFBNet: Learning Pseudo-Saliency Distribution via Feedback Convolutional Networks}, 
      author={Guanqun Ding and Nevrez Imamoglu and Ali Caglayan and Masahiro Murakawa and Ryosuke Nakamura},
      year={2021},
      eprint={2112.03731},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{ding2021fbnet,
  title={FBNet: FeedBack-Recursive CNN for Saliency Detection},
  author={Ding, Guanqun and {\.I}mamo{\u{g}}lu, Nevrez and Caglayan, Ali and Murakawa, Masahiro and Nakamura, Ryosuke},
  booktitle={2021 17th International Conference on Machine Vision and Applications (MVA)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```

## Getting Started
### 1. Installation
You can install the envs mannually by following commands:
```
git clone https://github.com/gqding/SalFBNet.git
conda create -n salfbnet python=3.8
conda activate salfbnet
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install scikit-learn scipy tensorboard tqdm
pip install torchSummaryX
```
Alternativaly, you can install the envs from yml file. Before running the command, please revise the 'prefix' with your PC name.
```
conda env create -f environment.yml
```

### 2. Downloads
- Our pre-trained models
    
    We released our pretrained SalFBNet models on [Google Drive](https://drive.google.com/drive/folders/1tUYgWPZvVn5k8xNZCSuv2lquNOSTzM7X?usp=sharing).

- Our Pseudo-Saliency dataset (~24.2GB)

    We released our PseudoSaliency dataset on [this page](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/PseudoSaliency.md).
        
- Our testing saliency results on public datasets

    You can download our testing saliency resutls from this [Google Drive](https://drive.google.com/drive/folders/1KwQhft1l_siVR33tFLUiihdwPcuA_3b6?usp=sharing).

### 3. Run
After downloading the pretrained models, you can run the script by 
```sh
sh run_test.sh
```
Alternativaly, you can modify the script for testing of different image folder and models (SalFBNet_Res18 or SalFBNet_Res18Fixed).
```sh
python main_test.py --model=pretrained_models/FBNet_Res18Fixed_best_model.pth \
--save_fold=./results_Res18Fixed/ \
--backbone=Res18Fixed\
--test_path=Datasets/PseudoSaliency/Images/ECSSD/images/
```
You can find results under the 'results_*' folder.
### 4. Datasets

Dataset | #Image | #Training | #Val. | #Testing | Size | URL | Paper
:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
[SALICON](http://salicon.net/challenge-2017/) | 20,000 | 10,000 | 5,000 | 5,000 | ~4GB | [download link](http://salicon.net/challenge-2017/) | [paper](http://www-users.cs.umn.edu/~qzhao/publications/pdf/salicon_cvpr15.pdf)
[MIT300](http://saliency.mit.edu/results_mit300.html) | 300 | - | - | 300 | ~44.4MB | [download link](http://saliency.mit.edu/BenchmarkIMAGES.zip) | [paper](https://dspace.mit.edu/handle/1721.1/68590)  
[MIT1003](http://people.csail.mit.edu/tjudd/WherePeopleLook/) | 1003 | 900* | 103* | - | ~178.7MB | [download link](http://people.csail.mit.edu/tjudd/WherePeopleLook/) | [paper](http://people.csail.mit.edu/tjudd/WherePeopleLook/Docs/wherepeoplelook.pdf)
[PASCAL-S](http://cbs.ic.gatech.edu/salobj/) | 850 | - | - | 850 | ~108.3MB | [download link](http://cbs.ic.gatech.edu/salobj/) | [paper](https://arxiv.org/pdf/1406.2807)
[DUT-OMRON](http://saliencydetection.net/dut-omron/) | 5,168 | - | - | 5,168 | ~151.8MB | [download link](http://saliencydetection.net/dut-omron/) | [paper](http://saliencydetection.net/dut-omron/download/manifold.pdf)
[TORONTO](http://www-sop.inria.fr/members/Neil.Bruce/) | 120 | - | - | 120 | ~12.3MB | [download link](http://www-sop.inria.fr/members/Neil.Bruce/eyetrackingdata.zip) | [paper](http://www-sop.inria.fr/members/Neil.Bruce/NIPS2005_0081.pdf)
[Pseudo-Saliency (Ours)](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/PseudoSaliency.md) | 176,880 | 150,000 | 26,880 | - | ~24.2GB | [download link](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/PseudoSaliency.md) | [paper](https://arxiv.org/abs/2112.03731)

* \*Training and Validation sets are randomly split by this work (see [Training list](https://github.com/gqding/SalFBNet/blob/main/Datasets/Eye_Fixation_Train_MIT1003.lst) and [Val list](https://github.com/gqding/SalFBNet/blob/main/Datasets/Eye_Fixation_Val_MIT1003.lst)).
* We released our Pseudo-Saliency dataset on this [link](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/PseudoSaliency.md).

## Performance Evaluation

### 1. Visulization Results
<img src="Figs/visualization.png" alt="input" style="width:600px">

### 2. Testing Performance on DUT-OMRON, PASCAL-S, and TORONTO
<img src="Figs/performance123.png" alt="input" style="width:600px">

### 3. Testing Performance on SALICON
<img src="Figs/performance4.png" alt="input" style="width:600px">

### 4. Testing Performance on MIT300
<img src="Figs/performance5.png" alt="input" style="width:600px">

Please check the leaderboard of [MIT300](https://saliency.tuebingen.ai/results.html) for more details.

Our SalFBNet model ranked in Second best with sAUC, CC, and SIM metrics (Screenshot from December 10, 2021).

<img src="Figs/mit300_leaderboard_sAUC_20211210.png" alt="input" style="width:600px">

### 5. Efficiency Comparison
<img src="Figs/efficiency.png" alt="input" style="width:600px">

## Pseudo-Saliency Dataset
### 1. Annotation
<img src="Figs/pseudo-saliency.png" alt="input" style="width:600px">

### 2. Pseudo Saliency Distribution
<img src="Figs/mean_sal.png" alt="input" style="width:600px">

### 3. PseudoSaliency Download
Downlaod link of [PseudoSaliency](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/PseudoSaliency.md) dataset can be found from [here](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/PseudoSaliency.md).

## Evaluation
We use the [metric implementation](https://github.com/cvzoya/saliency/tree/master/code_forMetrics) from MIT Saliency Benchmark for performance evaluation.

## Acknowledgement
- Data Collection

    We collect color images from [ImageNet](https://image-net.org/challenges/LSVRC/2012/index.php) and SOD datasets including [CSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [MSRA-B](https://mmcheng.net/msra10k/), [MSRA10K](https://mmcheng.net/msra10k/), [THUR15K](https://mmcheng.net/gsal/). 

- Pseudo-Annotators

    We use 5 models as our Pseudo-Annotators, including [DeepGazeIIE](https://github.com/matthias-k/DeepGaze), [UNISAL](https://github.com/rdroste/unisal), [MSINet](https://github.com/alexanderkroner/saliency), [EMLNet](https://github.com/SenJia/EML-NET-Saliency), [CASNetII](https://ncript.comp.nus.edu.sg/site/ncript-top/emotionalattention/).

    We appriciate their public datasets and codes.


