#!/usr/bin/env bash

base_dir="save_models/VOC_second"

# base class training
python train.py --dataset pascal_voc_0712 \
--epochs 21 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--meta_type 2 --meta_train True --meta_loss True

# bs 代表 batch size，nw 代表加载数据要多少个 worker
# meta_type = 1 or 2 or 3，代表三种划分方式