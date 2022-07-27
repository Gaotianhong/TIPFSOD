import os, sys
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import pandas as pd

import datasets
import datasets.custom
from .custom_eval import custom_eval
from datasets.imdb import imdb
from model.utils.config import cfg


class custom(imdb):
    def __init__(self, image_set, data_path, csv_file='custom_dataset.txt'):
        imdb.__init__(self, 'custom_{}'.format(image_set))
        self._image_set = image_set
        self._data_path = data_path
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

        df = pd.read_csv(os.path.join(data_path, csv_file))

        self.df = df[df.set == self._image_set]

        # Add novel classes after base classes [base / novel]
        self._classes = tuple(['__background__'] +
                              [c for c in np.unique(df.cls).tolist() if c not in cfg.CUSTOM_NOVEL_CLASSES] +
                              [c for c in np.unique(df.cls).tolist() if c in cfg.CUSTOM_NOVEL_CLASSES])
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        # The image index is set to be the unique image path in the dataset
        self._image_index = np.unique(self.df.im_path).tolist()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup': False}

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of pascal3d.
        """

        objs = self.df[self.df.im_path == index]
        num_objs = len(objs)

        # original annotation for object detection
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object annotation into a data frame.
        for ix in range(num_objs):
            x1 = max(float(objs.iloc[ix]['left']), 0)
            y1 = max(float(objs.iloc[ix]['upper']), 0)
            x2 = min(float(objs.iloc[ix]['right']), objs.iloc[ix]['width'] - 1)
            y2 = min(float(objs.iloc[ix]['lower']), objs.iloc[ix]['height'] - 1)
            cls = self._class_to_ind[objs.iloc[ix]['cls']]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

            ishards[ix] = objs.iloc[ix]['difficult']
            if cls not in self._classes:
                continue
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_results_file_template(self):
        # data_path/results/<comp_id>_det_test_aeroplane.txt
        filename = self._comp_id + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._data_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} results file'.format(cls))
            filename = self._get_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            ap = custom_eval(filename, self.df, self._image_set, cls, cachedir, ovthresh=0.5)
            print('AP for {} = {:.3f}'.format(cls, ap))

            aps.append(ap)

            if i == self.num_classes - len(cfg.CUSTOM_NOVEL_CLASSES):
                print('Mean AP = {:.4f} for base'.format(np.mean(aps)))
            if i == self.num_classes:
                print('Mean AP = {:.4f} for novel'.format(np.mean(aps[-20:])))

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        return np.mean(aps)

    def evaluate_detections(self, all_boxes, output_dir, **kwargs):
        self._write_results_file(all_boxes)
        AP = self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_results_file_template().format(cls)
                os.remove(filename)
        return AP