# --------------------------------------------------------
# Pytorch Meta R-CNN
# Written by Anny Xu, Xiaopeng Yan, based on the code from Jianwei Yang
# --------------------------------------------------------
import os
import os.path
import sys
import torch.utils.data as data
import cv2
import torch
import random
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from model.utils.config import cfg
import collections


class MetaDataset(data.Dataset):
    """Meta Dataset
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val')
        metaclass(string): the class name
        img_size(int) : the PRN network input size
        shot(int): the number of instances
        shuffle(bool)
    """

    def __init__(self, root, image_sets, metaclass, img_size, shots=1, shuffle=False, phase=1):
        self.root = root
        self.image_set = image_sets
        self.img_size = img_size
        self.metaclass = metaclass  # 15个基类 或 15个基类 + 5个新类
        self.shots = shots
        if phase == 2:
            self.shots = shots * 3
        self.shuffle = shuffle
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')  # 标注框路径
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')  # 图像路径
        self.shot_path = open(os.path.join(self.root, 'VOC2007', 'ImageSets/Main/shots.txt'), 'w')  # shot路径
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

        # 类别转换为 index 索引
        class_to_idx = dict(zip(self.metaclass, range(len(self.metaclass))))  # class to index mapping

        self.prndata = []
        self.prncls = []
        prn_image, prn_mask = self.get_prndata()
        for i in range(shots):
            cls = []
            data = []
            for n, key in enumerate(list(prn_image.keys())):
                img = torch.from_numpy(np.array(prn_image[key][i]))  # 图像
                img = img.unsqueeze(0)  # 读取图像
                mask = torch.from_numpy(np.array(prn_mask[key][i]))  # 掩码
                # if np.random.random() <= 0.25:  # 0.5 or 0.25
                #     mask = torch.from_numpy(np.array(prn_mask[key][i]))  # 掩码
                # else:
                #     mask = torch.ones(prn_mask[key][i].shape)  # 半监督场景，相当于无标记
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(3)
                imgmask = torch.cat([img, mask], dim=3)
                cls.append(class_to_idx[key])
                data.append(imgmask.permute(0, 3, 1, 2).contiguous())  # permute 使得维度换位 使得数据连续
            self.prncls.append(cls)  # 类别
            self.prndata.append(torch.cat(data, dim=0))  # 图像和掩码

    def __getitem__(self, index):
        return self.prndata[index], self.prncls[index]

    def get_prndata(self):
        '''
        :return: the construct prn input data
        :prn_image: lists of images in shape of (H, W, 3)
        :prn_mask: lists of masks in shape pf (H, W)
        '''
        if self.shuffle:
            random.shuffle(self.ids)
        prn_image = collections.defaultdict(list)  # image
        prn_mask = collections.defaultdict(list)  # bbox 掩码
        classes = collections.defaultdict(int)  # 类别
        for cls in self.metaclass:
            classes[cls] = 0  # 类别初始化
        for img_id in self.ids:
            target = ET.parse(self._annopath % img_id).getroot()  # 读取标注信息
            img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)  # 读取图像
            img = img.astype(np.float32, copy=False)
            img -= cfg.PIXEL_MEANS
            height, width, _ = img.shape
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)  # 读取的 mask 掩码
            h, w, _ = img.shape
            y_ration = float(h) / self.img_size
            x_ration = float(w) / self.img_size
            img_resize = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)  # 读取的 image
            for obj in target.iter('object'):  # 遍历图像中的物体类别
                difficult = int(obj.find('difficult').text) == 1
                if difficult:
                    continue
                name = obj.find('name').text.strip()
                if name not in self.metaclass:
                    continue
                if classes[name] >= self.shots:  # 每个类别选取K张图像
                    break
                classes[name] += 1
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(float(bbox.find(pt).text)) - 1
                    if i % 2 == 0:
                        cur_pt = int(cur_pt / x_ration)
                        bndbox.append(cur_pt)
                    elif i % 2 == 1:
                        cur_pt = int(cur_pt / y_ration)
                        bndbox.append(cur_pt)
                mask[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]] = 1  # mask 掩码
                prn_image[name].append(img_resize)  # image
                prn_mask[name].append(mask)  # mask 掩码
                self.shot_path.write(str(img_id[1]) + '\n')
                break
            if len(classes) > 0 and min(classes.values()) == self.shots:
                break
        self.shot_path.close()
        return prn_image, prn_mask

    def __len__(self):
        return len(self.prndata)
