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
        self.scene_per_episode_max = 10
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
                print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
                return roidb
        except FileNotFoundError as e:
            print(str(e))
            gt_roidb = [self._load_pedestrian_annotation(index)
                        for index in self.image_index]
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
            return gt_roidb

    def _load_pedestrian_annotation(self, index):

        [i, fid, set_name, video_name] = index.split("/")[-4:]

        # Load data from a data frame
        unit_frame = self.annotations[set_name][video_name]['frames'][fid][0]
        pos = np.array(unit_frame['pos'], dtype=np.float)
        label = unit_frame['lbl']
        # Make pixel indexes 0-based
        l = pos[0] - 1
        t = pos[1] - 1
        w = pos[2]
        h = pos[3]

        # num_obj = 1 only considering one person
        num_objs = 1
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pedestrian is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)


        # all indexes are 0 in this dataset
        ix = num_objs - 1
        # boxes, gt_classes, seg_areas, ishards, overlaps
        boxes[ix, :] = [l, t, l + w, t + h]
        cls = self._class_to_ind[label]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = w * h
        diffc = unit_frame['occl']
        difficult = 0 if diffc == None else diffc
        ishards[ix] = difficult
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
        # Example path to image set file:
        # CaltechPedestrians/images/
        image_set_file = self.image_path + "/ref.txt"
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

        if (self.area_thresh is not None) or (self.area_thresh != 0):
            print("Area Threshold exists for CaltechPedestrians dataset as {}".format(self.area_thresh))

        processed_image_index = []

        episodes = self.get_epsiode()
        # unit is tuple (set_name, video_name, str, end)
        i = 1
        for key, fids in episodes.items():
            set_name, video_name = key[:2]
            indices = np.sort(npr.choice(len(fids),self.scene_per_episode_max, replace=False)) \
                if len(fids) > self.scene_per_episode_max else np.arange(len(fids))
            for fid in np.asarray(fids)[indices]:
                unit_frame = self.annotations[set_name][video_name]['frames'][fid][0]
                pos = unit_frame['pos']
                label = unit_frame['lbl']
                area = float(pos[2]) * float(pos[3])
                # take label == 'person' / areas > 200.0
                if label != 'person' or area < self.area_thresh: continue
                else:
                    index = '{}/{}/{}/{}'.format(i, fid, set_name, video_name)
                    processed_image_index.append(index)
                    i += 1
        print('CaltechPedestrians dataset has {} images in total, Max per episode {} images'\
                                    .format(i, self.scene_per_episode_max))
        # this method takes all images (# 91199) as input
        # with open(image_set_file) as f:
        #     for x in f.readlines():
        #         index = x.strip()
        #         [i, fid, set_name, video_name] = index.split("/")[-4:]
        #         unit_frame = self.annotations[set_name][video_name]['frames'][fid][0]
        #         pos = unit_frame['pos']
        #         label = unit_frame['lbl']
        #         area = float(pos[2]) * float(pos[3])
        #         # take label == 'person' / areas > 200.0
        #         if label != 'person' or area < self.area_thresh: continue
        #         else: processed_image_index.append(index)

        return processed_image_index

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