from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import time
import cv2
import pickle
import torch
from torch.autograd import Variable

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_label_only

from matplotlib import pyplot as plt
import torch.utils.data as Data
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test TIP')
    # Define Model and data
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset:coco2017,coco,pascal_07_12',
                        default='pascal_07_12', type=str)  # 数据集 dest 为解析后的参数名称
    parser.add_argument('--net', dest='net',
                        help='metarcnn',
                        default='metarcnn', type=str)  # 网络结构
    # Define testing parameters
    parser.add_argument('--cuda', dest='cuda',
                        default=True, type=bool,
                        help='whether use CUDA')  # 是否使用 cuda
    parser.add_argument('--cag', dest='class_agnostic',
                        default=False, type=bool,  # 是否执行类无关的bbox回归 对绿框进行校正和精修来生成一个新的检测框
                        help='whether perform class_agnostic bbox regression')
    # Define meta parameters
    parser.add_argument('--meta_test', dest='meta_test', default=False, type=bool,
                        help='whether perform meta testing')  # meta testing
    parser.add_argument('--meta_loss', dest='meta_loss', default=False, type=bool,
                        help='whether perform adding meta loss')  # meta loss 是否增加
    parser.add_argument('--shots', dest='shots',
                        help='the number of meta input',
                        default=1, type=int)  # K shots 标记框
    parser.add_argument('--meta_type', dest='meta_type', default=1, type=int,
                        help='choose which sets of metaclass')  # meta_type 1, 2, 3
    parser.add_argument('--phase', dest='phase',
                        help='the phase of training process',
                        default=1, type=int)  # 阶段使用 train（1）或 finetune（2）
    # Define TGC Loss and TFMC Loss
    parser.add_argument('--TGC', dest='TGC', default=False, type=bool,
                        help='whether perform TGC')  # 是否引入 transformed guidance consistency(TGC) loss
    parser.add_argument('--TFMC', dest='TFMC', default=False, type=bool,
                        help='whether perform TFMC')  # 是否引入 transformed feature map consistency(TGC) loss
    # resume trained model
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="exps",
                        type=str)  # 训练好的模型
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=3256, type=int)  # shots
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=12, type=int)  # 第 epoch 代数加载模型
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=21985, type=int)
    # Others
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)  # batch size
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')  # 是否可视化
    parser.add_argument('--save', dest='save_dir',
                        help='directory to save logs', default='models',
                        type=str)  # 保存模型的路径
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE  # 学习率
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY  # 学习率递减

if __name__ == '__main__':
    args = parse_args()

    if args.net == 'metarcnn':
        from model.faster_rcnn.resnet import resnet
    print('Called with args:')
    print(args)
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "pascal_voc_0712":
        args.imdbval_name = "voc_2007_test"  # 测试集
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "object3d":
        args.imdbval_name = "objectnet3d_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.25, 0.5, 1, 2, 4]',
                         'MAX_NUM_GT_BOXES', '50']

    elif args.dataset == "custom":
        args.imdbval_name = "custom_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5, 1, 2]', 'MAX_NUM_GT_BOXES', '50']

    # the number of sets of metaclass
    cfg.TRAIN.META_TYPE = args.meta_type  # first second third
    args.cfg_file = "cfgs/res101_ms.yml"
    if args.cfg_file is not None:  # res101
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:  # fast-rcnn
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False  # 不使用水平翻转
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    input_dir = args.load_dir  # 保存模型的路径 save_models
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             '{}_{}_{}_{}.pth'.format(args.dataset, str(args.net), args.checksession,
                                                      args.checkepoch))
    # initilize the network here.
    if args.net == 'metarcnn':
        num_layers = 101 if args.dataset == 'pascal_voc_0712' else 50

        num_cls = imdb.num_classes

        fasterRCNN = resnet(num_cls, num_layers, pretrained=True, class_agnostic=args.class_agnostic,
                            meta_train=False, meta_test=args.meta_test, meta_loss=args.meta_loss)
    else:
        print('No module define')

    load_name = os.path.join(input_dir,
                             '{}_{}_{}_{}.pth'.format(args.dataset, str(args.net), args.checksession, args.checkepoch))
    fasterRCNN.create_architecture()
    print("load checkpoint %s" % load_name)  # 经过 train + finetune 的模型
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'], False)
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    if args.cuda:
        cfg.CUDA = True

    if args.cuda:  # 使用cuda
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis
    if vis:
        thresh = 0.5
    else:
        thresh = 0.0001

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    # if meta test
    mean_class_attentions = None
    if args.meta_test:
        print('loading mean class attentions!')  # 训练阶段的mean class attentions
        mean_class_attentions = pickle.load(open(os.path.join(
            input_dir, 'meta_type_{}'.format(args.meta_type),
            str(args.phase) + '_shots_' + str(args.shots) + '_mean_class_attentions.pkl'), 'rb'))

    num_images = len(imdb.image_index)  # 图像数量
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_cls)]

    output_dir = os.path.join(input_dir, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载数据
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             num_cls, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')  #

    with torch.no_grad():
        for i in range(num_images):  # 测试集总数 PASCAL VOC 4952
            data = next(data_iter)
            im_data_list = []  # 图像
            im_info_list = []  # 信息
            gt_boxes_list = []  # 标注框
            num_boxes_list = []  # 标注框个数
            # initilize the tensor holder here.
            im_data = torch.FloatTensor(1)
            im_info = torch.FloatTensor(1)
            num_boxes = torch.LongTensor(1)
            gt_boxes = torch.FloatTensor(1)
            # ship to cuda
            if args.cuda:
                im_data = im_data.cuda()
                im_info = im_info.cuda()
                num_boxes = num_boxes.cuda()
                gt_boxes = gt_boxes.cuda()
            # make variable
            im_data = Variable(im_data)  # 数据声明为Variable
            im_info = Variable(im_info)
            num_boxes = Variable(num_boxes)
            gt_boxes = Variable(gt_boxes)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            im_data_list.append(im_data)  # 转换为列表存储
            im_info_list.append(im_info)
            gt_boxes_list.append(gt_boxes)
            num_boxes_list.append(num_boxes)
            det_tic = time.time()
            # 测试时的损失函数
            rois, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, cls_prob_list, bbox_pred_list, _, _, _ = fasterRCNN(im_data_list, im_info_list,
                                                                            gt_boxes_list,
                                                                            num_boxes_list,
                                                                            mean_class_attentions=mean_class_attentions)
            if args.meta_test:  # 元学习测试
                for clsidx in range(len(cls_prob_list)):
                    cls_prob = cls_prob_list[clsidx]
                    bbox_pred = bbox_pred_list[clsidx]
                    scores = cls_prob.data
                    boxes = rois.data[:, :, 1:5]
                    if cfg.TEST.BBOX_REG:  # 默认为True
                        # Apply bounding-box regression deltas
                        box_deltas = bbox_pred.data
                        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:  # 默认为True
                            # Optionally normalize targets by a precomputed mean and stdev
                            if args.class_agnostic:  # 类无关的bbox回归
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4)
                            else:
                                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                                box_deltas = box_deltas.view(1, -1, 4 * num_cls)

                        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)  # 预测出的box
                        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

                    else:
                        # Simply repeat the boxes, once for each class
                        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                    pred_boxes /= im_info[0][2]
                    scores = scores.squeeze()
                    pred_boxes = pred_boxes.squeeze()
                    if clsidx == 0:
                        allscores = scores[:, clsidx].unsqueeze(1)
                        allpredboxes = pred_boxes if args.class_agnostic else pred_boxes[:,
                                                                              (clsidx) * 4:(clsidx + 1) * 4]

                        allscores = torch.cat([allscores, scores[:, (clsidx + 1)].unsqueeze(1)], dim=1)
                        allpredboxes = torch.cat([allpredboxes, pred_boxes], dim=1) if args.class_agnostic else \
                            torch.cat([allpredboxes, pred_boxes[:, (clsidx + 1) * 4:(clsidx + 2) * 4]], dim=1)
                    else:
                        allscores = torch.cat([allscores, scores[:, (clsidx + 1)].unsqueeze(1)], dim=1)
                        allpredboxes = torch.cat([allpredboxes, pred_boxes], dim=1) if args.class_agnostic else \
                            torch.cat([allpredboxes, pred_boxes[:, (clsidx + 1) * 4:(clsidx + 2) * 4]], dim=1)

                scores = allscores
                pred_boxes = allpredboxes
            else:  # 未使用元学习
                scores = cls_prob_list.data
                boxes = rois.data[:, :, 1:5]
                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred_list.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4 * num_cls)

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))
                pred_boxes /= data[1][0][2]
                scores = scores.squeeze()

            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic  # 检测时间
            misc_tic = time.time()
            if vis:  # 可视化
                im = cv2.imread(imdb.image_path_from_index(imdb.image_index[i]))
                im2show = np.copy(im)
            for j in range(1, num_cls):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)  # 非极大值抑制Å
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis:  # 画框
                        # im2show = vis_detections_label_only(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)  # 概率
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_cls)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, num_cls):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic  # 非极大值抑制时间

            # 动态打印
            sys.stdout.write(
                'im_detect: {:d}/{:d} {:.3f}s {:.3f}s  \r'.format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                im_dir = 'vis/' + str(data[4].numpy()[0]) + '_det.png'
                cv2.imwrite(im_dir, im2show)
                plt.imshow(im2show[:, :, ::-1])
                plt.show()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    ## comment following block for a direct print on screen
    ## ====================================================
    orig_stdout = sys.stdout
    f = open(os.path.join(output_dir, '{}shots_out.txt'.format(args.shots)), 'w')
    sys.stdout = f
    ## ====================================================

    # mkdir results 打印输出到文件
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir, **vars(args))
    end = time.time()
    print("test time: %0.4fs" % (end - start))
