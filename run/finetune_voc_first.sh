#!/usr/bin/env bash

base_dir="save_models/VOC_first"

# number of shots
for j in 1 2 3 5 10
do
# few-shot fine-tuning
python train.py --dataset pascal_voc_0712 \
--epochs 30 --bs 4 --nw 8 \
--log_dir checkpoint --save_dir $base_dir \
--r True --checksession 200 --checkepoch 20 \
--meta_type 1 --shots $j --phase 2 --meta_train True --meta_loss True --TGC True --TFMC True
done

# r代表恢复checkpoint，checksession代表段？，checkepoch预训练模型开始代数
# phase 第二阶段