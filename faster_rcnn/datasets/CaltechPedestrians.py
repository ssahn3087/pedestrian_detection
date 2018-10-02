import os
import json
import glob
import pickle
import scipy.sparse
import numpy as np
import random
from PIL import Image
import numpy.random as npr
from faster_rcnn.datasets.imdb import imdb
from faster_rcnn.fast_rcnn.config import cfg
from collections import defaultdict

class CaltechPedestrians(imdb):

    def __init__(self, image_set):
        imdb.__init__(self,  'CaltechPedestrians_' + image_set)
        self.triplet = True if 'triplet' in image_set.split('_') else False
        self._prefix = 'CaltechPedestrians'
        self.image_set = 'train' if 'train' in image_set.split('_') else 'test'
        self.num_triplet_set = 8000 if self.image_set == 'train' else 10000
        self.num_triplet_test_images = 3
        # object image condition ex) bbox of object is too small to recognize
        self.area_thresh = 200.0
        self.scene_per_episode_max = 15 if image_set == 'train' else 5
        self.image_path = os.path.join(cfg.DATA_DIR, self._prefix, "images")
        self.annotations_path = os.path.join(cfg.DATA_DIR, self._prefix, "annotations")
        self.annotations_file_name = "annotations.json"
        self.annotations = self._load_json_file()
        self.base_id_dict, self.total_id = self.get_base_id_dict()
        self._image_index_default = self._load_image_set_index()
        self._image_ext = '.jpg'
        self._classes = ('__background__',  # always index 0 but no background data
                         'person')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._roidb_handler = self.gt_roidb
        self._image_index = self.update_image_index()

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

        if self.image_set == 'train':
            cache_file = os.path.join(self.cache_path, self._prefix + '_train_gt_roidb.pkl')
        else:
            cache_file = os.path.join(self.cache_path, self._prefix + '_test_gt_roidb.pkl')
        try:
            if os.path.getsize(cache_file) > 0:
                with open(cache_file, 'rb') as fid:
                    roidb = pickle.load(fid)
                assert len(roidb) == len(self.image_index), \
                    'fatal error: the length of _image_index must be same with roidb'
                print('{} gt roidb loaded from {}'.format(self.name, cache_file))
        except FileNotFoundError as e:
            print(str(e))
            roidb = [self._load_pedestrian_annotation(index)
                        for index in self._image_index_default]
            roidb = self.update_image_index_file(roidb)
            with open(cache_file, 'wb') as fid:
                pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        finally:
            if self.triplet:
                if self.image_set == 'train':
                    cache_file = os.path.join(self.cache_path, self._prefix + '_triplet_train_gt_roidb.pkl')
                else:
                    cache_file = os.path.join(self.cache_path, self._prefix + '_triplet_test_gt_roidb.pkl')
                try:
                    if os.path.getsize(cache_file) > 0:
                        with open(cache_file, 'rb') as fid:
                            roidb = pickle.load(fid)
                    self._image_index = self.update_image_index(triplet=self.triplet)
                    print('{} gt roidb loaded from {}'.format(self.name, cache_file))
                except FileNotFoundError as e:
                    print(str(e))
                    if self.image_set == 'train':
                        roidb = self.make_triplet_set(self.gather_by_id(roidb))
                    else:
                        roidb = self.make_triplet_test_set(self.gather_by_id(roidb))
                    with open(cache_file, 'wb') as fid:
                        pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
                    self.update_image_index_file(roidb, triplet=self.triplet)
                    print('wrote gt roidb to {}'.format(cache_file))
            else:
                roidb = self.gather_by_episode(roidb)
            print('CaltechPedestrians dataset for {} : {} images loaded'.format(self.image_set,len(roidb)))
            return roidb

    def update_image_index(self, triplet=False):
        if triplet:
            image_index_file = os.path.join(self.cache_path, self.image_set + "_triplet.txt")
        else:
            image_index_file = os.path.join(self.cache_path, self.image_set + ".txt")
        f = open(image_index_file, 'r')
        lines = f.readlines()
        image_index = [line.strip() for line in lines]
        f.close()
        return image_index

    def update_image_index_file(self, gt_roidb, triplet=False):
        if triplet:
            image_index_file = os.path.join(self.cache_path, self.image_set + "_triplet.txt")
            with open(image_index_file, 'w') as f:
                for i in range(len(self._image_index)):
                    f.write(self._image_index[i] + '\n')
            roidb = gt_roidb
        else:
            roidb = [db for db in gt_roidb if db is not None]
            image_index =[]
            image_index_file = os.path.join(self.cache_path, self.image_set + ".txt")
            with open(image_index_file, 'w') as f:
                for i, db in enumerate(gt_roidb):
                    if db is not None:
                        image_index.append(self._image_index_default[i])
                        f.write(self._image_index_default[i]+'\n')
            self._image_index = image_index

        assert len(self._image_index) == len(roidb), \
            'fatal error: the length of _image_index must be same with roidb'

        return roidb

    def _load_pedestrian_annotation(self, index):

        [set_name, video_name, fid, i] = index.split("/")[-4:]

        # Load data from a data frame
        objs = self.annotations[set_name][video_name]['frames'][fid]

        # Check abnormal data and Remove
        objs = [obj for obj in objs if self.object_condition_satisfied(obj)]

        num_objs = len(objs)
        if num_objs == 0:
            return None
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pedestrian is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)
        ids = np.zeros((num_objs), dtype=np.uint16)
        for ix, obj in enumerate(objs):
            pos = np.round(np.array(obj['pos'], dtype=np.float32))
            label = obj['lbl']

            # let negative position be at 0
            indices = np.where(pos[:2] < 1)[0]
            pos[indices] = 1

            # Make pixel indexes 0-based
            l = pos[0] - 1
            t = pos[1] - 1
            w = pos[2]
            h = pos[3]

            img = self.image_path_from_index(index)
            (width, height) = Image.open(img).size
            # boxes, gt_classes, seg_areas, ishards, overlaps
            boxes[ix, :2] = [l, t]
            boxes[ix, 2] = width - 1 if l + w >= width else l + w
            boxes[ix, 3] = height - 1 if t + h >= height else t + h

            cls = self._class_to_ind[label]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = w * h
            ids[ix] = obj['id'] + self.base_id_dict[set_name, video_name]
            # diffc = unit_frame['occl']
            # difficult = 0 if diffc == None else diffc
            ishards[ix] = 0
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas,
                'ids': ids}

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
        [set_name, video_name, fid, i] = index.split("/")[-4:]
        prefix = os.path.join(self.image_path, self.image_set)
        image_path = "{}/{}/{}/{}{}".format(prefix, set_name, video_name, fid, self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self.image_path, self.image_set, self.image_set + ".txt")
        f = open(image_set_file, 'r')
        lines = f.readlines()
        image_index = [line.strip() for line in lines]
        f.close()
        return image_index

    def object_condition_satisfied(self, obj):
        pos = np.array(obj['pos'], dtype=np.float32)
        label = obj['lbl']
        occl = int(obj['occl'])
        area = float(pos[2] * pos[3])
        # take label == 'person' / areas > 200.0
        if self.image_set == 'test':
            occl = 0
            self.area_thresh = 0

        if label != 'person' or area < self.area_thresh or occl == 1:
            return False
        elif (pos[3:] <= 10).any() or pos.size < 4:
            return False
        else:
            return True

    def get_epsiode(self):
        episodes = defaultdict(list)
        prefix = os.path.join(self.image_path, self.image_set)
        for set_path in sorted(glob.glob(prefix + '/set*')):
            set_name = set_path.split("/")[-1]
            for video_path in sorted(glob.glob(set_path + '/V*')):
                video_name = video_path.split("/")[-1]
                unit = self.annotations[set_name][video_name]["frames"]
                for fid, v in unit.items():
                    for i, obj in enumerate(unit[fid]):
                        _str = unit[fid][i]['str']
                        _end = unit[fid][i]['end']
                        episodes[(set_name, video_name, _str, _end)].append(fid)
        print("CaltechPedestrians dataset consists of {} episodes".format(len(episodes)))
        return episodes

    def gather_by_episode(self, roidb):
        episodes = self.get_epsiode()
        image_index = ['/'.join(index.split('/')[:-1]) for index in self.image_index]
        new_index = []
        for keys, fids in episodes.items():
            set_name, video_name = keys[:2]
            indices = np.sort(npr.choice(len(fids), self.scene_per_episode_max, replace=False)) \
                if len(fids) > self.scene_per_episode_max else np.arange(len(fids))
            for fid in np.asarray(fids)[indices]:
                index = '{}/{}/{}'.format(set_name, video_name, fid)
                new_index.append(index)
        new_index = np.asarray(list(set(new_index)))
        image_index = np.asarray(image_index)
        match_indices = np.where(np.isin(image_index, new_index))[0]
        roidb = np.array(roidb)[match_indices].tolist()
        self._image_index = np.array(self._image_index)[match_indices].tolist()
        assert len(self._image_index) == len(roidb), \
            'fatal error: the length of _image_index must be same with roidb'
        print('CaltechPedestrians dataset has {} images in total, Max per episode {} images' \
              .format(len(roidb), self.scene_per_episode_max))
        return roidb

    def get_base_id_dict(self):
        anns = self.annotations
        base_id_dict = {}
        base_id = 0
        for set_name in anns.keys():
            for video_name in anns[set_name].keys():
                ids = []
                for frame in anns[set_name][video_name]['frames'].values():
                    for obj in frame:
                        if obj['id'] not in ids:
                            ids.append(obj['id'])
                base_id_dict[set_name, video_name] = base_id
                base_id += max(ids) + 1 if len(ids) != 0 else 0
                if set_name == list(anns.keys())[-1] and video_name == list(anns[set_name].keys())[-1]:
                    total_id = base_id
        return base_id_dict, total_id

    def gather_by_id(self, roidb):
        image_index = self._image_index
        new_index = []
        new_roidb = []
        for i, db in enumerate(roidb):
            n_obj = len(db['boxes'])

            for ix in range(n_obj):
                unit_db = {}
                for k in db.keys():
                    if isinstance(db[k], (np.ndarray, np.generic)):
                        unit_db[k] = np.expand_dims(db[k][ix], axis=0)
                    elif isinstance(db[k], scipy.sparse.csr_matrix):
                        unit_db[k] = scipy.sparse.csr_matrix(\
                            np.expand_dims(db[k].toarray()[ix], axis=0))
                    else:
                        unit_db[k] = db[k]
                new_roidb.append(unit_db)
                new_index.append(image_index[i])
        self._image_index = new_index
        return new_roidb

    def make_triplet_set(self, roidb):
        for i in range(len(self._image_index)):
            roidb[i]['index'] = self._image_index[i]

        id_roidb = {db['ids'][0]: [] for db in roidb}
        for db in roidb:
            id_roidb[db['ids'][0]] += [db]
        for k in list(id_roidb.keys()):
            if len(id_roidb[k]) < 2:
                del id_roidb[k]

        exist_ids = list(id_roidb.keys())
        exist_ids.sort()
        identical_different = []
        far = 1
        while far > 0:
            for i in range(len(exist_ids) - far):
                if len(identical_different) < self.num_triplet_set:
                    identical_different.append([exist_ids[i], exist_ids[i + far]])
                else:
                    far = -1
                    break
            far += 1

            # call = random.sample(exist_ids, 2)
            # if call not in identical_different:
            #     identical_different.append(call)
        random.shuffle(identical_different)
        new_index = []
        new_roidb = []
        for twin in identical_different:
            pos = random.sample(id_roidb[twin[0]], 2)
            neg = random.sample(id_roidb[twin[1]], 1)
            new_index.extend([pos[0]['index'], pos[1]['index'], neg[0]['index']])
            new_roidb.extend([pos[0], pos[1], neg[0]])
        for db in new_roidb:
            db.pop('index', None)
        self._image_index = new_index
        return new_roidb

    def make_triplet_test_set(self, roidb, shuffle=True):
        for i in range(len(self._image_index)):
            roidb[i]['index'] = self._image_index[i]

        id_roidb = {db['ids'][0]: [] for db in roidb}

        for db in roidb:
            id_roidb[db['ids'][0]] += [db]
        for k in list(id_roidb.keys()):
            if len(id_roidb[k]) < 2:
                del id_roidb[k]
        exist_ids = list(id_roidb.keys())
        exist_ids.sort()
        identical_different = []
        for t in range(len(exist_ids) - self.num_triplet_test_images + 2):
            call = []
            for i in range(self.num_triplet_test_images-1):
                call.append(exist_ids[t+i])
            identical_different.append(call)
        while len(identical_different) < self.num_triplet_set:
            call = random.sample(exist_ids, self.num_triplet_test_images - 1)
            if call not in identical_different:
                identical_different.append(call)
        random.shuffle(identical_different)
        new_index = []
        new_roidb = []
        for sib in identical_different:
            idn = random.choice(sib) if shuffle else sib[0]
            sib.remove(idn)
            all = []
            all.extend(random.sample(id_roidb[idn], 2))
            all.extend([random.sample(id_roidb[diff], 1)[0] for diff in sib])
            random.shuffle(all)
            new_index.extend([db['index'] for db in all])
            new_roidb.extend([db for db in all])
        for db in new_roidb:
            db.pop('index', None)
        self._image_index = new_index
        return new_roidb