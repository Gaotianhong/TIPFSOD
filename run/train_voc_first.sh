#!/usr/bin/env bash

base_dir="save_models/VOC_first"

# base class training
python train.py --dataset pascal_voc_0712 \
--epochs 21 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--meta_type 1 --meta_train True --meta_loss True

# epochs迭代次数，bs代表batch size
# meta_type元学习类型（1，2，3），使用meta_train和meta_loss