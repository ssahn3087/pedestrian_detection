import os
import json
import glob
import pickle as cPickle
import scipy.sparse
import numpy as np
from .imdb import imdb
from ..fast_rcnn.config import cfg


class CaltechPedestrians(imdb):
    def __init__(self, name):
        imdb.__init__(self, name)
        self.image_path = os.path.join(cfg.DATA_DIR, self._name, "images")
        self.annotations_path = os.path.join(cfg.DATA_DIR, self._name, "annotations")
        self._annotations = self._load_json_file()
        self._image_ext = '.jpg'
        self._classes = ('__background__',  # always index 0 but no background data
                         'person')
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb

    """
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
                        'str' : start frame where object appears
                        'end' : end frame where object disappears
                        'lbl' : label (class)
                        'hide' : object is hiden
                        'init' : object exists 1/0}], ... }
    """

    def _load_json_file(self):
        with open(self.annotations_path+"/annotations.json") as json_file:
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
                    roidb = cPickle.load(fid)
                print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
                return roidb
        except FileNotFoundError as e:
            print(str(e))
            gt_roidb = self._load_pedestrian_annotation()
            with open(cache_file, 'wb') as fid:
                for db in gt_roidb:
                    cPickle.dump(db, fid, cPickle.HIGHEST_PROTOCOL)
                print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_pedestrian_annotation(self):
        gt_roidb = []

        for set_path in sorted(glob.glob(self.image_path+'/set*')):
            set_name = set_path.split("/")[-1]
            for video_path in sorted(glob.glob(set_path+'/V*')):
                video_name = video_path.split("/")[-1]
                all_frames = self._annotations[set_name][video_name]["frames"]
                for fid, val in all_frames.items():
                    # num_obj = 1 only considering one person
                    num_objs = 1
                    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
                    gt_classes = np.zeros((num_objs), dtype=np.int32)
                    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
                    # "Seg" area for pedestrian is just the box area
                    seg_areas = np.zeros((num_objs), dtype=np.float32)
                    ishards = np.zeros((num_objs), dtype=np.int32)

                    # Load data from a data frame
                    unit_frame = all_frames[fid][0]
                    # all indexes are 0 in this dataset
                    ix = num_objs - 1
                    pos = unit_frame['pos']

                    # Make pixel indexes 0-based
                    l = float(pos[0]) - 1
                    t = float(pos[1]) - 1
                    w = float(pos[0])
                    h = float(pos[1])
                    boxes[ix, :] = [l, t, l + w, t + h]
                    gt_classes[ix] = 1
                    overlaps[ix, 1] = 1.0
                    seg_areas[ix] = w * h

                    diffc = unit_frame['occl']
                    difficult = 0 if diffc == None else diffc
                    ishards[ix] = difficult

                    overlaps = scipy.sparse.csr_matrix(overlaps)

                    gt_roidb.append({'boxes': boxes,
                            'gt_classes': gt_classes,
                            'gt_ishard': ishards,
                            'gt_overlaps': overlaps,
                            'flipped': False,
                            'seg_areas': seg_areas,
                            'set_name' : set_name, 'video_name' : video_name})
        return gt_roidb

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
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
