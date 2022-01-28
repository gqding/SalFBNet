# PseudoSaliency Dataset

![](https://img.shields.io/static/v1?label=dataset&message=PseudoSaliency&color=blue)
![](https://img.shields.io/static/v1?label=size&message=24.2GB&color=<COLOR>)
![](https://img.shields.io/static/v1?label=image-number&message=176,880&color=<COLOR>)
![](https://img.shields.io/static/v1?label=label-number&message=176,880&color=<COLOR>)
![](https://img.shields.io/static/v1?label=training&message=150,000&color=<COLOR>)
![](https://img.shields.io/static/v1?label=validation&message=26,880&color=<COLOR>)

This is a guide for PseudoSaliency Dataset. This dataset has been published on the following paper:

SalFBNet: Learning Pseudo-Saliency Distribution via Feedback Convolutional Networks, 2021. ([pdf](https://arxiv.org/pdf/2112.03731.pdf))

Guanqun Ding, Nevrez Imamoglu, Ali Caglayan, Masahiro Murakawa, Ryosuke Nakamura

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

### Downloads
1. Download annotated maps of PseudoSaliency saliency dataset, and unzip the file by:
```
wget PseudoSaliency_avg_release.zip
unzip PseudoSaliency_avg_release.zip
```
2. Download [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html) dataset and put the file "HKU-IS.rar" to the folder "PseudoSaliency_avg_release/".
3. Download [MSRA10K](https://mmcheng.net/msra10k/) dataset and put the file "MSRA10K_Imgs_GT.zip" to the folder "PseudoSaliency_avg_release/".
4. Download [MSRA-B](https://mmcheng.net/msra10k/) dataset and put the file "MSRA-B.zip" to the folder "PseudoSaliency_avg_release/".
5. Download [THUR15K](https://mmcheng.net/code-data/) dataset and put the file "THUR15000.zip" to the folder "PseudoSaliency_avg_release/".
6. Download [CSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) dataset and put the file "images.zip" to the folder "PseudoSaliency_avg_release/CSSD/".
7. Download [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) dataset and put the file "images.zip" to the folder "PseudoSaliency_avg_release/ECSSD/".
8. You might be asked to register an account and login for obtaining the ImageNet download webpage.
9. Find the "ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)"
10. Download [ImageNet](https://image-net.org/download.php) dataset from the officail [website](https://image-net.org/download.php). following files:
- [Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622], ILSVRC2012_img_val.tar
- [Test images (all tasks). 13GB. MD5: e1b8681fff3d63731c599df9b4b6fc02"], ILSVRC2012_img_test_v10102019.tar
11. Assume that all compressed files have been downloaded under the folder "PseudoSaliency_avg_release/", and these files are organized by follows:
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
5. Uncompress all files
```
cd CSSD/
unzip images.zip
cd ..

cd ECSSD/
unzip images.zip
cd ..

unrar x HKU-IS.rar

mkdir ILSVRC2012_img_val
tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val

mkdir ILSVRC2012_img_test_v10102019
tar -xvf ILSVRC2012_img_test_v10102019.tar -C ILSVRC2012_img_test_v10102019

unzip MSRA-B.zip

unzip MSRA10K_Imgs_GT.zip

unzip THUR15000.zip

```
