# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
# --------------------------------------------------------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import pickle


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, n_classes, class_agnostic, meta_train, meta_test=None, meta_loss=None, TGC_loss=None,
                 TFMC_loss=None):
        super(_fasterRCNN, self).__init__()
        self.n_classes = n_classes
        self.class_agnostic = class_agnostic
        self.meta_train = meta_train
        self.meta_test = meta_test
        self.meta_loss = meta_loss
        self.TGC_loss = TGC_loss
        self.TFMC_loss = TFMC_loss

        # loss
        self.RCNN_loss_cls = 0  # ResNet101
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    # 继承nn.module
    def forward(self, im_data_list, im_info_list, gt_boxes_list, num_boxes_list,
                average_shot=None, mean_class_attentions=None):
        # return attentions for testing
        if average_shot:
            prn_data = im_data_list[0]  # len(metaclass)*4*224*224
            attentions = self.prn_network(prn_data)
            return attentions
        # extract attentions for training 支持集 Support Set
        if self.meta_train and self.training:
            prn_data = im_data_list[0]  # len(metaclass)*4*224*224
            # feed prn data to prn_network
            attentions = self.prn_network(prn_data)
            prn_cls = im_info_list[0]  # len(metaclass)

        im_data = im_data_list[-1]  # Query Set
        im_info = im_info_list[-1]
        gt_boxes = gt_boxes_list[-1]
        num_boxes = num_boxes_list[-1]

        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map，ResNet 101 相当于 BackBone
        base_feat = self.RCNN_base(self.rcnn_conv1(im_data))  # 原始图像和标注信息产生 feature map

        # 创造与矩阵匹配的单通道高斯噪声
        def create_gaussian_noise(img):
            noise = torch.cuda.FloatTensor(img.shape)
            noise = torch.randn(img.shape, out=noise)
            return noise

        # 噪声权重
        gaussian_noise_weight = torch.nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=True)
        gaussian_noise_weight.data.fill_(30)

        # 加入高斯噪声 TFMC Loss
        im_data += gaussian_noise_weight * create_gaussian_noise(im_data)
        transformed_base_feat = self.RCNN_base(self.rcnn_conv1(im_data))

        # feed base feature map tp RPN to obtain rois，产生感兴趣区域，RPN损失 Query Set
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(transformed_base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':  # 默认是align，原始图像base_feat和转换图像RoI对齐
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))  # (b*128)*1024*7*7
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  # (b*128)*2048

        # meta training phase Support Set
        if self.meta_train:
            rcnn_loss_cls = []
            rcnn_loss_bbox = []

            # pooled feature maps need to operate channel-wise multiplication with
            # the corresponding class's attentions of every roi of image
            for b in range(batch_size):
                zero = Variable(torch.FloatTensor([0]).cuda())
                proposal_labels = rois_label[b * 128:(b + 1) * 128].data.cpu().numpy()[0]
                unique_labels = list(np.unique(proposal_labels))  # the unique rois labels of the input image

                for i in range(attentions.size(0)):  # attentions len(attentions)*2048
                    if prn_cls[i].numpy()[0] + 1 not in unique_labels:
                        rcnn_loss_cls.append(zero)  # ResNet loss
                        rcnn_loss_bbox.append(zero)
                        continue

                    roi_feat = pooled_feat[b * cfg.TRAIN.BATCH_SIZE:(b + 1) * cfg.TRAIN.BATCH_SIZE, :]  # 128*2048
                    cls_feat = attentions[i].view(1, -1, 1, 1)  # 1*2048*1*1

                    diff_feat = roi_feat - cls_feat.squeeze()
                    corr_feat = F.conv2d(roi_feat.unsqueeze(-1).unsqueeze(-1),
                                         cls_feat.permute(1, 0, 2, 3),
                                         groups=2048).squeeze()

                    # subtraction + correlation: [bs, 2048]
                    channel_wise_feat = torch.cat((self.corr_fc(corr_feat), self.diff_fc(diff_feat)), dim=1)

                    # combined with the roi feature: [bs, 2048 * 2]
                    channel_wise_feat = torch.cat((channel_wise_feat, roi_feat), dim=1)

                    # compute object bounding box regression
                    bbox_pred = self.RCNN_bbox_pred(channel_wise_feat)  # 128*4
                    if self.training and not self.class_agnostic:
                        # select the corresponding columns according to roi labels
                        bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                        batch_rois_label = rois_label[b * cfg.TRAIN.BATCH_SIZE:(b + 1) * cfg.TRAIN.BATCH_SIZE]
                        bbox_pred_select = torch.gather(
                            bbox_pred_view, 1, batch_rois_label.view(
                                batch_rois_label.size(0), 1, 1).expand(batch_rois_label.size(0), 1, 4))
                        bbox_pred = bbox_pred_select.squeeze(1)

                    # compute object classification probability
                    cls_score = self.RCNN_cls_score(channel_wise_feat)

                    if self.training:
                        # classification loss
                        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label[b * 128:(b + 1) * 128])
                        rcnn_loss_cls.append(RCNN_loss_cls)  # L_Q_CLS
                        # bounding box regression L1 loss
                        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target[b * 128:(b + 1) * 128],
                                                         rois_inside_ws[b * 128:(b + 1) * 128],
                                                         rois_outside_ws[b * 128:(b + 1) * 128])  # L_LOC

                        rcnn_loss_bbox.append(RCNN_loss_bbox)

            # meta attentions loss
            if self.meta_loss:
                attentions_score = self.Meta_cls_score(attentions)
                # torch.cat 代表按行拼接
                meta_loss = F.cross_entropy(attentions_score, Variable(torch.cat(prn_cls, dim=0).cuda()))  # L_Q_CLS
            else:
                meta_loss = 0

            if self.TGC_loss:  # Support Set
                # L2 范数
                base_feat_pro = self.prn_network(prn_data)  # 图像 + 标注框 得到引导向量
                prn_data_copy = prn_data.clone()  # 拷贝
                prn_data_copy += gaussian_noise_weight * create_gaussian_noise(prn_data_copy)  # 转换 module
                transformed_base_feat_pro = self.prn_network(prn_data_copy)  # 转换图像 + 标注框
                L2_loss = torch.nn.MSELoss(reduce=True, size_average=True)
                tgc_loss = L2_loss(base_feat_pro, transformed_base_feat_pro.detach())  # 转换前后图像计算 TGC loss
            else:
                tgc_loss = 0

            if self.TFMC_loss:  # 查询集 Query Set feature map loss
                # L2 范数
                L2_loss = torch.nn.MSELoss(reduce=True, size_average=True)
                tfmc_loss = L2_loss(base_feat, transformed_base_feat.detach())  # Query Set 转换前后图像计算 TFMC loss
            else:
                tfmc_loss = 0

            return rois, rpn_loss_cls, rpn_loss_bbox, rcnn_loss_cls, rcnn_loss_bbox, rois_label, 0, 0, meta_loss, tgc_loss, tfmc_loss

        # meta testing phase
        elif self.meta_test:
            cls_prob_list = []
            bbox_pred_list = []
            for i in range(len(mean_class_attentions)):
                mean_attentions = mean_class_attentions[i]

                cls_feat = mean_attentions.view(1, -1, 1, 1)  # 1*2048*1*1

                diff_feat = pooled_feat - cls_feat.squeeze()
                corr_feat = F.conv2d(pooled_feat.unsqueeze(-1).unsqueeze(-1),
                                     cls_feat.permute(1, 0, 2, 3),
                                     groups=2048).squeeze()

                # subtraction + correlation: [bs, 2048]
                channel_wise_feat = torch.cat((self.corr_fc(corr_feat), self.diff_fc(diff_feat)), dim=1)

                # combined with the roi feature: [bs, 2048 * 2]
                channel_wise_feat = torch.cat((channel_wise_feat, pooled_feat), dim=1)

                # compute bbox offset
                bbox_pred = self.RCNN_bbox_pred(channel_wise_feat)
                if self.training and not self.class_agnostic:
                    # select the corresponding columns according to roi labels
                    bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(
                        bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                    bbox_pred = bbox_pred_select.squeeze(1)

                # compute object classification probability
                cls_score = self.RCNN_cls_score(channel_wise_feat)
                cls_prob = F.softmax(cls_score)

                RCNN_loss_cls = 0
                RCNN_loss_bbox = 0

                cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
                bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
                cls_prob_list.append(cls_prob)
                bbox_pred_list.append(bbox_pred)

            return rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob_list, bbox_pred_list, 0, 0, 0

        # original faster-rcnn implementation
        else:
            bbox_pred = self.RCNN_bbox_pred(pooled_feat)
            if self.training and not self.class_agnostic:
                # select the corresponding columns according to roi labels
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(
                    bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

            # compute object classification probability
            cls_score = self.RCNN_cls_score(pooled_feat)  # 128 * 1001
            cls_prob = F.softmax(cls_score)

            RCNN_loss_cls = 0
            RCNN_loss_bbox = 0

            if self.training:
                # classification loss
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
            bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, cls_prob, bbox_pred, 0, 0, 0

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def weights_normal_init(model, dev=0.001):
            if isinstance(model, list):
                for m in model:
                    weights_normal_init(m, dev)
            else:
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        m.weight.data.normal_(0.0, dev)
                    elif isinstance(m, nn.Linear):
                        m.weight.data.normal_(0.0, dev)
                    elif isinstance(m, torch.nn.BatchNorm1d):
                        m.weight.data.normal_(1.0, 0.02)
                        m.bias.data.fill_(0)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        weights_normal_init(self.corr_fc)
        weights_normal_init(self.diff_fc)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
