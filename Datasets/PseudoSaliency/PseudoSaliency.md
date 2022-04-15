# SalFBNet Pseudo Saliency Dataset

![](https://img.shields.io/static/v1?label=dataset&message=PseudoSaliency&color=blue)
![](https://img.shields.io/static/v1?label=size&message=24.2GB&color=<COLOR>)
![](https://img.shields.io/static/v1?label=image-number&message=176,880&color=<COLOR>)
![](https://img.shields.io/static/v1?label=label-number&message=176,880&color=<COLOR>)
![](https://img.shields.io/static/v1?label=training&message=150,000&color=<COLOR>)
![](https://img.shields.io/static/v1?label=validation&message=26,880&color=<COLOR>)

This is a guide for PseudoSaliency Dataset. This dataset has been published on the following paper:

SalFBNet: Learning Pseudo-Saliency Distribution via Feedback Convolutional Networks, 2021. ([arXiv](https://arxiv.org/abs/2112.03731))

Guanqun Ding, Nevrez Imamoglu, Ali Caglayan, Masahiro Murakawa, Ryosuke Nakamura

### Citation
Please cite the following papers if you use our data or codes in your research.

```
@article{ding2022salfbnet,
  title={SalFBNet: Learning pseudo-saliency distribution via feedback convolutional networks},
  author={Ding, Guanqun and {\.I}mamo{\u{g}}lu, Nevrez and Caglayan, Ali and Murakawa, Masahiro and Nakamura, Ryosuke},
  journal={Image and Vision Computing},
  pages={104395},
  year={2022},
  publisher={Elsevier}
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

## Pseudo-Saliency Dataset
### 1. Annotation
<img src="https://github.com/gqding/SalFBNet/blob/main/Figs/pseudo-saliency.png" alt="input" style="width:600px">

### 2. Pseudo Saliency Distribution
<img src="https://github.com/gqding/SalFBNet/blob/main/Figs/mean_sal.png" alt="input" style="width:600px">

### Downloads
1. Download annotated maps of [**PseudoSaliency saliency dataset**](https://salfbnet-pseudo-saliency-dataset.s3.abci.ai/PseudoSaliency_avg_release.zip), and unzip the file by:
```sh
wget https://salfbnet-pseudo-saliency-dataset.s3.abci.ai/PseudoSaliency_avg_release.zip
unzip PseudoSaliency_avg_release.zip
```
- *Please note that we only provide annotated maps as the copyright for the images does not belong to us.*
- *Please download the images by following steps 2-8.*

2. Download [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html) dataset and put the file "HKU-IS.rar" to the folder "PseudoSaliency_avg_release/".
3. Download [MSRA10K](https://mmcheng.net/msra10k/) dataset and put the file "MSRA10K_Imgs_GT.zip" to the folder "PseudoSaliency_avg_release/".
4. Download [MSRA-B](https://mmcheng.net/msra10k/) dataset and put the file "MSRA-B.zip" to the folder "PseudoSaliency_avg_release/".
5. Download [THUR15K](https://mmcheng.net/code-data/) dataset and put the file "THUR15000.zip" to the folder "PseudoSaliency_avg_release/".
6. Download [CSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) dataset and put the file "images.zip" to the folder "PseudoSaliency_avg_release/CSSD/".
7. Download [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) dataset and put the file "images.zip" to the folder "PseudoSaliency_avg_release/ECSSD/".
8. Download [ImageNet](https://image-net.org/download.php) dataset from the officail [website](https://image-net.org/download.php). You might be asked to register an account and login for obtaining the ImageNet download links. Find the "ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)" and download these two files ("ILSVRC2012_img_val.tar" and "ILSVRC2012_img_test_v10102019.tar"):
```
Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622
Test images (all tasks). 13GB. MD5: e1b8681fff3d63731c599df9b4b6fc02" 
```
9. Make sure that all folders and files are organized by follows:
```
PseudoSaliency_avg_release/  
├── HKU-IS.rar
├── ILSVRC2012_img_test_v10102019.tar
├── ILSVRC2012_img_val.tar
├── MSRA10K_Imgs_GT.zip
├── MSRA-B.zip
└── THUR15000.zip  
├── Images
│   ├── CSSD
│   │   └── images
        └── images.zip
│   ├── ECSSD
│   │   └── images
    │   └── images.zip
│   ├── HKU-IS
│   │   ├── gt
│   │   └── imgs
│   ├── ILSVRC2012_img_test_v10102019
│   │   └── test
│   ├── ILSVRC2012_img_val
│   ├── MSRA10K_Imgs_GT
│   │   └── Imgs
│   ├── MSRA-B
│   └── THUR15000
│       ├── Butterfly
│       │   └── Src
│       ├── CoffeeMug
│       │   └── Src
│       ├── DogJump
│       │   └── Src
│       ├── Giraffe
│       │   └── Src
│       └── plane
│           └── Src
└── Maps
    ├── CSSD
    │   └── images
    ├── ECSSD
    │   └── images
    ├── HKU-IS
    │   └── imgs
    ├── ILSVRC2012_img_test_v10102019
    │   └── test
    ├── ILSVRC2012_img_val
    ├── MSRA10K_Imgs_GT
    │   └── Imgs
    ├── MSRA-B
    └── THUR15000
        ├── Butterfly
        │   └── Src
        ├── CoffeeMug
        │   └── Src
        ├── DogJump
        │   └── Src
        ├── Giraffe
        │   └── Src
        └── plane
            └── Src
```
10. Extrac all compressed files by following commands:
```sh
cd PseudoSaliency_avg_release/CSSD/
unzip images.zip

cd PseudoSaliency_avg_release/ECSSD/
unzip images.zip

cd PseudoSaliency_avg_release/
unrar x HKU-IS.rar

cd PseudoSaliency_avg_release/
tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val

cd PseudoSaliency_avg_release/
tar -xvf ILSVRC2012_img_test_v10102019.tar -C ILSVRC2012_img_test_v10102019

cd PseudoSaliency_avg_release/
unzip MSRA-B.zip

cd PseudoSaliency_avg_release/
unzip MSRA10K_Imgs_GT.zip

cd PseudoSaliency_avg_release/
unzip THUR15000.zip
```
11. Use the script "[demo_check_all_files.py](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/demo_check_all_files.py)" to check that all files exist.  
```sh
python demo_check_all_files.py
```

### Data Description
There are two text files that hold a list of paths of the training and validation sets for the PseudoSaliency dataset.

For each line in the text file, the contents are
```
<image_path label_path>
```
For example,
```
Images/MSRA10K_Imgs_GT/Imgs/177838.jpg Maps/MSRA10K_Imgs_GT/Imgs/177838.jpg
```
You can easily read all path lists with following script (see [demo_check_all_files.py](https://github.com/gqding/SalFBNet/blob/main/Datasets/PseudoSaliency/demo_check_all_files.py))
```python
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

img_root = "./"
filename = os.path.join(img_root, "Train_List_PseudoSaliency.lst")
lines = [line.rstrip('\n') for line in open(filename)]
image_paths = list(map(lambda x: os.path.join(img_root, x.split(' ')[0]), lines))
label_paths = list(map(lambda x: os.path.join(img_root, x.split(' ')[1]), lines))
print("Starting to check all images and labels of training set...")
read_image=True
for img_path, lb_path in tqdm(zip(image_paths, label_paths), desc="training set", total=len(image_paths)):
    assert os.path.exists(img_path) and os.path.exists(lb_path)
    if read_image:
        image = cv2.imread(img_path)
        label = cv2.imread(lb_path, 0)
```

## Acknowledgement
- Data Collection

    We collect color images from [ImageNet](https://image-net.org/challenges/LSVRC/2012/index.php) and SOD datasets including [CSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [MSRA-B](https://mmcheng.net/msra10k/), [MSRA10K](https://mmcheng.net/msra10k/), [THUR15K](https://mmcheng.net/gsal/). 

- Pseudo-Annotators

    We use 5 models as our Pseudo-Annotators, including [DeepGazeIIE](https://github.com/matthias-k/DeepGaze), [UNISAL](https://github.com/rdroste/unisal), [MSINet](https://github.com/alexanderkroner/saliency), [EMLNet](https://github.com/SenJia/EML-NET-Saliency), [CASNetII](https://ncript.comp.nus.edu.sg/site/ncript-top/emotionalattention/).

    We appriciate their public datasets and codes.
