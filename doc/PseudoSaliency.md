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

### Download
1. Download annotated maps of PseudoSaliency saliency dataset, and unzip the file:
```
wget PseudoSaliency_avg_release.zip
unzip PseudoSaliency_avg_release.zip
```
3. You might be asked to register an account and login for obtaining the ImageNet download webpage.
4. Find the "ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)"
5. Download the following files:
- [Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622], ILSVRC2012_img_val.tar
- [Test images (all tasks). 13GB. MD5: e1b8681fff3d63731c599df9b4b6fc02"], ILSVRC2012_img_test_v10102019.tar
6. Assume that all compressed files have been downloaded under the folder "PseudoSaliency_avg_release/", and these files are organized by follows:
```
downloaded_files/
├── CSSD
│   └── images.zip
├── ECSSD
│   └── images.zip
├── HKU-IS.rar
├── ILSVRC2012_img_test_v10102019.tar
├── ILSVRC2012_img_val.tar
├── MSRA10K_Imgs_GT.zip
├── MSRA-B.zip
└── THUR15000.zip   
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

```
