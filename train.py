import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import random

from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import torch.utils.data as Data
from roi_data_layer.roidb import combined_roidb, rank_roidb_ratio, filter_class_roidb_flip, clean_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.resnet import resnet
import pickle

from datasets.metadata import MetaDataset
from datasets.metadata_coco import MetaDatasetCOCO
from datasets.metadata_TFA import MetaDatasetTFA

from datasets.metadata_3d import MetaDataset3D
from datasets.custom_metadata import MetaDatasetCustom
from collections import OrderedDict


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train TIP')
    # Define training data and Model 数据集和模型
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset:coco2017,coco,pascal_07_12',
                        default='pascal_voc_0712', type=str)
    parser.add_argument('--net', dest='net',
                        help='metarcnn',
                        default='metarcnn', type=str)  # metarcnn网络
    parser.add_argument('--TFA', default=False, type=bool,
                        help='use TFA split')  # coco数据集
    # Define display and save dir
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)  # 从第 1 代开始迭代
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=21, type=int)  # 迭代次数
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)  # 在一个迭代中间隔多少个 batch 显示
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)  # 每多少个迭代显示
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./models",
                        type=str)  # 模型保存路径
    # Define training parameters 训练参数
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)  # 加载数据要多少个worker
    parser.add_argument('--cuda', dest='cuda', default=True, type=bool,
                        help='whether use CUDA')  # 是否使用 cuda GPU
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)  # batch_size 即一次训练所抓取的数据样本数
    parser.add_argument('--cag', dest='class_agnostic', default=False, type=bool,
                        # 是否执行类无关的bbox回归 对绿框进行校正和精修来生成一个新的检测框
                        help='whether perform class_agnostic bbox regression')
    # Define meta parameters
    parser.add_argument('--meta_train', dest='meta_train', default=False, type=bool,
                        help='whether perform meta training')  # 是否使用元学习
    parser.add_argument('--meta_loss', dest='meta_loss', default=False, type=bool,
                        help='whether perform adding meta loss')  # 是否增加元学习损失

    parser.add_argument('--fix_encoder', action='store_true',
                        help='whether fix feature extraction')  # 特征提取

    parser.add_argument('--phase', dest='phase',
                        help='the phase of training process',
                        default=1, type=int)  # 训练过程的阶段，划分为两个阶段（train和finetune）
    parser.add_argument('--shots', dest='shots',
                        help='the number meta input of PRN network',
                        default=1, type=int)  # shots 的数量，k-shots 即标注信息的数量
    parser.add_argument('--meta_type', dest='meta_type', default=0, type=int,
                        help='choose which sets of metaclass')  # 选择哪一类对元学习（1，2，3）PASCAL VOC
    # Define TGC Loss and TFMC Loss
    parser.add_argument('--TGC', dest='TGC', default=False, type=bool,
                        help='whether perform TGC')  # 是否引入 transformed guidance consistency(TGC) loss
    parser.add_argument('--TFMC', dest='TFMC', default=False, type=bool,
                        help='whether perform TFMC')  # 是否引入 transformed feature map consistency(TGC) loss
    # config optimization 优化器
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)  # 训练优化器，梯度下降法
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)  # learning rate 初始学习率
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=4, type=int)  # 学习率递减
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)  # 学习率下降率
    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)  # 针对多 gpu 情形
    # resume trained model 使用已训练好的模型
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)  # 恢复checkpoint
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)  # 检查 session 以加载模型
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',  # 检查 epoch 以加载模型
                        default=10, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=21985, type=int)  # checkpoint 以加载模型
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)  # tensorboard展示
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to save logs', default='logs',
                        type=str)
    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size  # 训练大小
        self.num_per_batch = int(train_size / batch_size)  # iteration 总数据 / 批处理大小
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()  # 得到0-batchsize序号
        self.leftover_flag = False
        if train_size % batch_size:  # 判断数据是否能够整除
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True  # 保留最后不能整除的批次数据

    def __iter__(self):
        # 将数据批次的标号打乱，并和数据批次的长度相乘，得到随机排序后的照片起始标号
        # 给定参数n，返回一个从0到n-1的随机整数排列
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        # 将起始位置扩充，得到具体每一张图片对应的标号
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        # 将所有标号打平铺开
        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            # 将不能整除的数据与之前的数据合并
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    # tensorflow tensorboard 展示面板
    if args.use_tfboard:
        writer = SummaryWriter(args.log_dir)

    # dataset configuration
    if args.dataset == "object3d":
        base_num = 80
        if args.phase == 1:
            args.imdb_name = "objectnet3d_train"
        else:
            args.imdb_name = "objectnet3d_shots"
        args.imdbval_name = "objectnet3d_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]',
                         'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "coco":  # MSCOCO 2014
        base_num = 60  # 基类
        if args.phase == 1:
            args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        else:
            if args.TFA:
                args.imdb_name = 'coco_2014_TFA{}shot'.format(args.shots)
            else:
                args.imdb_name = "coco_2014_shots"
        args.imdbval_name = "coco_2014_minival"  # 测试集
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "pascal_voc_0712":  # PASCAL VOC
        base_num = 15  # 15个基本类
        if args.phase == 1:  # phase为训练阶段，train
            if args.meta_type == 1:  # first
                args.imdb_name = "voc_2007_train_first_split+voc_2012_train_first_split"
            elif args.meta_type == 2:  # second
                args.imdb_name = "voc_2007_train_second_split+voc_2012_train_second_split"
            elif args.meta_type == 3:  # third
                args.imdb_name = "voc_2007_train_third_split+voc_2012_train_third_split"
        else:
            args.imdb_name = "voc_2007_shots"  # finetune
        args.imdbval_name = "voc_2007_test"  # 测试集
        # Faster R-CNN 参数，ANCHOR_SCALES 基础尺寸的缩放比例，ANCHOR_RATIOS 长宽比，MAX_NUM_GT_BOXES bbox 数量
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "custom":
        base_num = 20  # modify this to your real base class number
        if args.phase == 1:
            args.imdb_name = "custom_train"
        else:
            args.imdb_name = "custom_shots"
        args.imdbval_name = "custom_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']

    # the number of sets of metaclass
    cfg.TRAIN.META_TYPE = args.meta_type  # first,second,third

    cfg.USE_GPU_NMS = args.cuda  # GPU
    if args.cuda:
        cfg.CUDA = True

    args.cfg_file = "cfgs/res101_ms.yml"  # res101 提取 guidance vector
    if args.cfg_file is not None:  # ResNet101
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:  # Faster R-CNN 参数
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if args.phase == 1:
        # First phase only use the base classes，support set 每个类别选择所有（200）标注样本
        shots = 200

        if args.meta_type == 1:  # use the first sets of base classes
            metaclass = cfg.TRAIN.BASECLASSES_FIRST  # 15 个基类
        if args.meta_type == 2:  # use the second sets of base classes
            metaclass = cfg.TRAIN.BASECLASSES_SECOND  # 15 个基类
        if args.meta_type == 3:  # use the third sets of base classes
            metaclass = cfg.TRAIN.BASECLASSES_THIRD  # 15 个基类
    else:
        # Second phase only use fewshot number of base and novel classes support set 每个类别选择所有K个标注样本
        shots = args.shots

        if args.meta_type == 1:  # use the first sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_FIRST  # 20 个类别 15 + 5
        if args.meta_type == 2:  # use the second sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_SECOND  # 20 个类别 15 + 5
        if args.meta_type == 3:  # use the third sets of all classes
            metaclass = cfg.TRAIN.ALLCLASSES_THIRD  # 20 个类别 15 + 5

    # prepare meta sets for meta training
    if args.meta_train:
        # construct the input dataset of PRN network
        img_size = 224

        if args.dataset == "coco":
            if args.TFA:
                metadataset = MetaDatasetTFA('data/coco', 'train', '2014', img_size, shots=shots)
            else:
                metadataset = MetaDatasetCOCO('data/coco', 'train', '2014', img_size,
                                              shots=shots, shuffle=True, phase=args.phase)
            metaclass = metadataset.metaclass

        elif args.dataset == 'pascal_voc_0712':
            if args.phase == 1:
                img_set = [('2007', 'trainval'), ('2012', 'trainval')]
            else:
                img_set = [('2007', 'trainval')]
            # metaclass 为first second third，随机选择 N 个类别，理解为 support set
            metadataset = MetaDataset('data/VOCdevkit', img_set, metaclass, img_size,
                                      shots=shots, shuffle=True, phase=args.phase)

        elif args.dataset == "object3d":
            metadataset = MetaDataset3D('/home/xiao/Datasets/ObjectNet3D', 'ObjectNet3D_new.txt', img_size, 'train',
                                        shots=shots, shuffle=True, phase=args.phase)
            metaclass = metadataset.metaclass

        elif args.dataset == "custom":
            metadataset = MetaDatasetCustom('Custom_Dataset_Pth', 'Custom.txt', img_size, 'train',
                                            shots=shots, shuffle=True, phase=args.phase)
            metaclass = metadataset.metaclass

        # metadata set 相当于 support set N-way K-Shot
        metaloader = torch.utils.data.DataLoader(metadataset, batch_size=1,
                                                 shuffle=False, num_workers=0, pin_memory=True)

    # imdb_name为数据集 imdb,roidb数据增强，roidb为训练集？
    # imdb为原始图像数据集，roidb为增强后的存放RoI，ratio_list返回升序排列的宽高比值，ratio_index为升序后的图像编号
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    # 根据 shot 从新类别中得到训练样本
    # filter out training samples from novel categories according to shot number
    print('\nbefore class filtering, there are %d images...' % (len(roidb)))
    if args.dataset != "pascal_voc_0712" and args.phase == 1:
        roidb = filter_class_roidb_flip(roidb, 0, imdb, base_num)
        ratio_list, ratio_index = rank_roidb_ratio(roidb)
        imdb.set_roidb(roidb)

    # filter roidb for the second phase 第二阶段
    if args.phase == 2 and not args.TFA:
        roidb = filter_class_roidb_flip(roidb, args.shots, imdb, base_num)
        ratio_list, ratio_index = rank_roidb_ratio(roidb)
        imdb.set_roidb(roidb)
    print('after class filtering, there are %d images...\n' % (len(roidb)))  # 类别过滤

    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))
    sys.stdout.flush()

    output_dir = args.save_dir  # 保存模型路径
    if not os.path.exists(output_dir):  # 创建目录
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    # 训练集
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    # 加载数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers, pin_memory=False)

    # initilize the network here 使用metarcnn网络
    if args.net == 'metarcnn':
        num_layers = 101 if args.dataset == 'pascal_voc_0712' else 50
        fasterRCNN = resnet(imdb.num_classes, num_layers, pretrained=True, class_agnostic=args.class_agnostic,
                            meta_train=args.meta_train, meta_loss=args.meta_loss,
                            tgc_loss=args.TGC, tfmc_loss=args.TFMC)
    fasterRCNN.create_architecture()  # ResNet101

    # initilize the optimizer here
    lr = cfg.TRAIN.LEARNING_RATE  # 0.001
    # lr = args.lr  # 学习率
    params = []
    # named_parameters 得到（模块名，模块参数）
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    if args.optimizer == "adam":  # 自适应矩估计
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":  # 梯度下降
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:  # output_dir（save_models） 预训练模型
        load_name = os.path.join(output_dir, '{}_metarcnn_{}_{}.pth'.format(
            args.dataset, args.checksession, args.checkepoch))  # checksession 为 shots
        print("loading checkpoint %s" % load_name)  # 输出加载模型的名称
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']  # 阶段
        args.start_epoch = checkpoint['epoch']  # 代数

        # the number of classes in second phase is different from first phase
        if args.phase == 2:  # 第二阶段
            new_state_dict = OrderedDict()
            # initilize params of RCNN_cls_score and RCNN_bbox_pred for second phase
            feat_dim = 2048 * 2
            RCNN_cls_score = nn.Linear(feat_dim, imdb.num_classes)  # 类别数量
            RCNN_bbox_pred = nn.Linear(feat_dim, 4) if args.class_agnostic else nn.Linear(feat_dim,
                                                                                          4 * imdb.num_classes)
            Meta_cls_score = nn.Linear(2048, imdb.num_classes)
            for k, v in checkpoint['model'].items():
                name = k
                new_state_dict[name] = v
                if 'RCNN_cls_score.weight' in k:
                    new_state_dict[name] = RCNN_cls_score.weight
                if 'RCNN_cls_score.bias' in k:
                    new_state_dict[name] = RCNN_cls_score.bias
                if 'RCNN_bbox_pred.weight' in k:
                    new_state_dict[name] = RCNN_bbox_pred.weight
                if 'RCNN_bbox_pred.bias' in k:
                    new_state_dict[name] = RCNN_bbox_pred.bias
                if 'Meta_cls_score.weight' in k:
                    new_state_dict[name] = Meta_cls_score.weight
                if 'Meta_cls_score.bias' in k:
                    new_state_dict[name] = Meta_cls_score.bias
            if args.TGC:
                new_state_dict["TGC_score.weight"] = Meta_cls_score.weight
                new_state_dict["TGC_score.bias"] = Meta_cls_score.bias
            if args.TFMC:
                new_state_dict["TFMC_score.weight"] = Meta_cls_score.weight
                new_state_dict["TFMC_score.bias"] = Meta_cls_score.bias
            fasterRCNN.load_state_dict(new_state_dict)

            # simple finetuning strategy used in Frustratingly simple few-shot object detection
            # 未使用
            if args.fix_encoder:
                print('fixing the feature extration modules')
                for p in fasterRCNN.meta_conv1.parameters():
                    p.requires_grad = False
                for p in fasterRCNN.rcnn_conv1.parameters():
                    p.requires_grad = False
                for p in fasterRCNN.RCNN_base.parameters():
                    p.requires_grad = False
                for p in fasterRCNN.RCNN_top.parameters():
                    p.requires_grad = False

        elif args.phase == 1:  # 第一阶段 已实现
            fasterRCNN.load_state_dict(checkpoint['model'])  # model
            optimizer.load_state_dict(checkpoint['optimizer'])  # optimizer
            lr = optimizer.param_groups[0]['lr']

        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % load_name)

    iters_per_epoch = int(train_size / args.batch_size)  # 每一个 epoch 迭代多少张图像

    # 开始训练
    for epoch in range(args.start_epoch, args.max_epochs):
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:  # 调整学习率
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)  # 数据迭代器 query set
        meta_iter = iter(metaloader)  # 元学习迭代器 support set
        for step in range(iters_per_epoch):
            try:
                data = next(data_iter)
            except:
                data_iter = iter(dataloader)
                data = next(data_iter)

            im_data_list = []  # 图像
            im_info_list = []  # 信息，类别特征
            gt_boxes_list = []  # bbox
            num_boxes_list = []  # box 个数

            # tensor 不能反向传播，Variable可以
            # initilize the tensor holder here.
            im_data = torch.FloatTensor(1)  # 张量
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)
            # ship to cuda
            if args.cuda:  # 转换到使用cuda
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()
            # make variable
            im_data = Variable(im_data)  # 变量
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)

            # get prn network input data 相当于 support set
            if args.meta_train:
                try:
                    prndata, prncls = next(meta_iter)
                except:
                    meta_iter = iter(metaloader)
                    prndata, prncls = next(meta_iter)

                im_data_list.append(Variable(prndata.squeeze(0).cuda()))  # squeeze降维
                im_info_list.append(prncls)  # 图像分类信息

            im_data.data.resize_(data[0].size()).copy_(data[0])  # query set
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            im_data_list.append(im_data)  # fasterRCNN参数以列表的显示传入
            im_info_list.append(im_info)  # 图像分类信息
            gt_boxes_list.append(gt_boxes)
            num_boxes_list.append(num_boxes)

            fasterRCNN.zero_grad()  # 零梯度

            # 损失函数 rpn_loss_cls rpn_loss_box RCNN_loss_cls RCNN_loss_bbox meta_loss
            rois, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob, bbox_pred, \
            meta_loss, tgc_loss, tfmc_loss = fasterRCNN(im_data_list, im_info_list, gt_boxes_list, num_boxes_list)

            # 计算损失函数
            if args.meta_train:
                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + sum(RCNN_loss_cls) / args.batch_size + sum(
                    RCNN_loss_bbox) / args.batch_size + meta_loss / len(metaclass) + tgc_loss / len(
                    metaclass) + tfmc_loss / len(metaclass)
            else:
                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                       + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            loss_temp += loss.data[0]

            # backward 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval  # loss_temp is aver loss

                loss_rpn_cls = rpn_loss_cls.item()  # rpn loss_cls 分类损失 RoI loss
                loss_rpn_box = rpn_loss_box.item()  # rpn loss_loc box回归 RoI loss
                if args.meta_train and not args.TGC:
                    loss_rcnn_cls = sum(RCNN_loss_cls) / args.batch_size
                    loss_rcnn_box = sum(RCNN_loss_bbox) / args.batch_size
                    loss_metarcnn = meta_loss / len(metaclass)  # loss_meta
                elif args.TGC:  # TGC 建立在 meta-learning 的基础上
                    loss_rcnn_box = sum(RCNN_loss_bbox) / args.batch_size  # L_LOC loss
                    loss_G_cls = meta_loss / len(metaclass)  # L_G_CLS loss
                    loss_rcnn_cls = sum(RCNN_loss_cls) / args.batch_size  # L_Q_CLS loss
                    loss_tgc = tgc_loss / len(metaclass)  # L_TGC loss
                    loss_tfmc = tfmc_loss / len(metaclass)  # L_TFMC loss
                else:
                    loss_rcnn_cls = RCNN_loss_cls.data[0]
                    loss_rcnn_box = RCNN_loss_bbox.data[0]

                # 通过 softmax 判断 anchors 属于 foreground 或 background
                fg_cnt = torch.sum(rois_label.data.ne(0))  # foreground
                bg_cnt = rois_label.data.numel() - fg_cnt  # background

                # lr 学习率
                print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                if args.meta_train and not args.TGC:
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, meta_loss %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_metarcnn))
                elif args.TGC and args.TFMC:
                    print(
                        "\t\t\trpn_cls: %.4f, rpn_box: %.4f, LOC_loss %.4f, G_cls_loss %.4f, Q_cls_loss %.4f, "
                        "TGC_loss %.4f, TFMC_loss %.4f"
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_box, loss_G_cls, loss_rcnn_cls, loss_tgc, loss_tfmc))
                else:
                    print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                          % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                sys.stdout.flush()

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    niter = (epoch - 1) * iters_per_epoch + step
                    for tag, value in info.items():
                        writer.add_scalar(tag, value, niter)

                loss_temp = 0
                start = time.time()

        if args.meta_train:
            save_name = os.path.join(output_dir,
                                     '{}_{}_{}_{}.pth'.format(str(args.dataset), str(args.net), shots, epoch))
        else:
            save_name = os.path.join(output_dir,
                                     '{}_{}_{}_{}.pth'.format(str(args.dataset), str(args.net), epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
        end = time.time()
        print(end - start)  # 运行时间

    # to extract the mean classes attentions of shots for testing
    if args.meta_train:  # 元学习
        with torch.no_grad():
            class_attentions = collections.defaultdict(list)
            meta_iter = iter(metaloader)
            for i in range(shots):
                prndata, prncls = next(meta_iter)
                im_data_list = []
                im_info_list = []
                gt_boxes_list = []
                num_boxes_list = []

                im_data = torch.FloatTensor(1)
                if args.cuda:
                    im_data = im_data.cuda()
                im_data = Variable(im_data)
                im_data.data.resize_(prndata.squeeze(0).size()).copy_(prndata.squeeze(0))
                im_data_list.append(im_data)

                # attentions
                attentions = fasterRCNN(im_data_list, im_info_list, gt_boxes_list, num_boxes_list, average_shot=True)
                for idx, cls in enumerate(prncls):
                    class_attentions[int(cls)].append(attentions[idx])

        # calculate mean class data of every class
        mean_class_attentions = {k: sum(v) / len(v) for k, v in class_attentions.items()}
        save_path = os.path.join(output_dir, 'meta_type_{}'.format(args.meta_type))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path,
                               str(args.phase) + '_shots_' + str(shots) + '_mean_class_attentions.pkl'), 'wb') as f:
            pickle.dump(mean_class_attentions, f, pickle.HIGHEST_PROTOCOL)
        print('save ' + str(args.shots) + ' mean classes attentions done!')
