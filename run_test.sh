#!/bin/bash

python main_test.py --model=pretrained_models/FBNet_Res18_best_model.pth \
 --save_fold=./results_Res18/ \
 --backbone=Res18\
 --test_path=Datasets/MIT300/BenchmarkIMAGES/

#python main_test.py --model=pretrained_models/FBNet_Res18Fixed_best_model.pth \
# --save_fold=./results_Res18Fixed/ \
# --backbone=Res18Fixed\
# --test_path=Datasets/MIT300/BenchmarkIMAGES/

#python main_test.py --model=pretrained_models/FBNet_Res18_best_model.pth \
# --save_fold=./results_Res18/ \
# --backbone=Res18\
# --test_path=Datasets/PseudoSaliency/Images/ECSSD/images/

#python main_test.py --model=pretrained_models/FBNet_Res18Fixed_best_model.pth \
# --save_fold=./results_Res18Fixed/ \
# --backbone=Res18Fixed\
# --test_path=Datasets/PseudoSaliency/Images/ECSSD/images/
