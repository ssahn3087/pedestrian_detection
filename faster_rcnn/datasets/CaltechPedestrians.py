import os
import json
import glob
import pickle
import scipy.sparse
import numpy as np
import numpy.random as npr
from faster_rcnn.datasets.imdb import imdb
from faster_rcnn.fast_rcnn.config import cfg
from collections import defaultdict

class CaltechPedestrians(imdb):

    def __init__(self, name):
        imdb.__init__(self, name)
        # object image condition ex) bbox of object is too small to recognize
        self.area_thresh = 200.0
        self.scene_per_episode_max = 15
        self.image_path = os.path.join(cfg.DATA_DIR, self._name, "images")
        self.annotations_path = os.path.join(cfg.DATA_DIR, self._name, "annotations")
        self.annotations_file_name = "annotations.json"
        self.annotations = self._load_json_file()
        self._image_ext = '.jpg'
        self._classes = ('__background__',  # always index 0 but no background data
                         'person')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

    """
    Dataset from : http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/
        --annotation file configure--
        ex) set_name = set00-010 / video_name = V000-014
        data[set_name][video_name]['nFrame']  = nFrame
        data[set_name][video_name]['maxObj']  = maxObj
        data[set_name][video_name]['log']     = log.tolist()
        data[set_name][video_name]['logLen']  = logLen
        data[set_name][video_name]['altered'] = altered
        data[set_name][video_name]['frames']  = defaultdict(list)
        
        --configure
        data[set_name][video_name]['frames'] = 
        {'frame_id' : [{'id' : object_id
                        'pos' : [ l t w h ]  (x_horizon_coord y_vertical_coord w h)
                        'occl' : occlusion
                        'lock' : object is fixed
                        'posv' : visible part, if posv ==[0, 0, 0, 0], entire object is visibile
                        'str' : start frame where the object appears
                        'end' : end frame where the object disappears
                        'lbl' : label (class)
                        'hide' : object is hiden
                        'init' : object exists 1/0}], ... }
    """

    def _load_json_file(self):
        with open(os.path.join(self.annotations_path, self.annotations_file_name)) as json_file:
            data = json.load(json_file)
        return data

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        try:
            if os.path.getsize(cache_file) > 0:
                with open(cache_file, 'rb') as fid:
                    roidb = pickle.load(fid)
                self.update_image_index(roidb)
                print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
                return roidb
        except FileNotFoundError as e:
            print(str(e))
            gt_roidb = [self._load_pedestrian_annotation(index)
                        for index in self.image_index]
            gt_roidb = self.remove_none(gt_roidb)
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
            return gt_roidb

    def update_image_index(self, roidb):
        image_set_file = self.image_path + "/ref.txt"
        f = open(image_set_file, 'r')
        lines = f.readlines()
        image_index = [line.strip() for line in lines]
        f.close()
        self._image_index = image_index
        print('CaltechPedestrians dataset has {} images in total, Max per episode {} images' \
              .format(len(roidb), self.scene_per_episode_max))
        assert len(self._image_index) == len(roidb), 'Create cache file again, ref.txt has been damaged'

    def remove_none(self, gt_roidb):
        roidb = [db for db in gt_roidb if db is not None]
        image_index =[]
        image_set_file = self.image_path + "/ref.txt"
        with open(image_set_file, 'w') as f:
            for i, db in enumerate(gt_roidb):
                if db is not None:
                    image_index.append(self._image_index[i])
                    f.write(self._image_index[i]+'\n')
        self._image_index = image_index
        print('CaltechPedestrians dataset has {} images in total, Max per episode {} images' \
              .format(len(roidb), self.scene_per_episode_max))
        assert len(self._image_index) == len(roidb), \
            'fatal error: the length of _image_index must be same with roidb'

        return roidb

    def _load_pedestrian_annotation(self, index):

        [i, fid, set_name, video_name] = index.split("/")[-4:]

        # Load data from a data frame
        objs = self.annotations[set_name][video_name]['frames'][fid]

        # Check abnormal data and Remove
        objs = [obj for obj in objs if self.object_condition_satisfied(obj, index)]

        num_objs = len(objs)
        if num_objs == 0:
            return None
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pedestrian is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)
        for ix, obj in enumerate(objs):
            pos = np.round(np.array(obj['pos'], dtype=np.float32))
            label = obj['lbl']
            # Make pixel indexes 0-based
            l = pos[0] - 1
            t = pos[1] - 1
            w = pos[2]
            h = pos[3]
            # boxes, gt_classes, seg_areas, ishards, overlaps
            boxes[ix, :] = [l, t, l + w, t + h]
            cls = self._class_to_ind[label]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = w * h
            # diffc = unit_frame['occl']
            # difficult = 0 if diffc == None else diffc
            ishards[ix] = 0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

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
        [i, fid, set_name, video_name] = index.split("/")[-4:]
        image_path = "{}/{}/{}/{}{}".format(self.image_path, set_name, video_name, fid, self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """

        if (self.area_thresh is not None) or (self.area_thresh != 0):
            print("Area Threshold exists for CaltechPedestrians dataset as {}".format(self.area_thresh))

        image_index = []

        episodes = self.get_epsiode()
        # unit is tuple (set_name, video_name, str, end)
        i = 1
        for key, fids in episodes.items():
            set_name, video_name = key[:2]
            indices = np.sort(npr.choice(len(fids),self.scene_per_episode_max, replace=False)) \
                if len(fids) > self.scene_per_episode_max else np.arange(len(fids))
            for fid in np.asarray(fids)[indices]:
                index = '{}/{}/{}/{}'.format(i, fid, set_name, video_name)
                image_index.append(index)
                i += 1

        return image_index

    def object_condition_satisfied(self, obj, index):
        from PIL import Image
        image_path = self.image_path_from_index(index)
        (width, height) = Image.open(image_path).size
        pos = np.round(np.array(obj['pos'], dtype=np.float32))
        label = obj['lbl']
        occl = int(obj['occl'])
        area = float(pos[2] * pos[3])
        # take label == 'person' / areas > 200.0
        if label != 'person' or area < self.area_thresh or occl == 1:
            return False
        elif pos[0] + pos[2] > width or pos[1] + pos[3] > height \
                or (pos < 0).any() or pos.size < 4:
            return False
        else:
            return True

    def get_epsiode(self):
        episodes = defaultdict(list)
        for set_path in sorted(glob.glob(self.image_path + '/set*')):
            set_name = set_path.split("/")[-1]
            for video_path in sorted(glob.glob(set_path + '/V*')):
                video_name = video_path.split("/")[-1]
                unit = self.annotations[set_name][video_name]["frames"]
                for fid, v in unit.items():
                    if fid == '0': continue
                    _str = unit[fid][0]['str']
                    _end = unit[fid][0]['end']
                    episodes[(set_name, video_name, _str, _end)].append(fid)
        print("CaltechPedestrians dataset consists of {} episodes".format(len(episodes)))
        return episodes