import os
import json
import numpy as np
from faster_rcnn.fast_rcnn.config import cfg
import cv2
import glob

DATA_DIR = cfg.DATA_DIR
dir_name = "CaltechPedestrians"
file_name = "annotations.json"

anno_file = os.path.join(DATA_DIR, dir_name, "annotations", file_name)
images_dir = os.path.join(DATA_DIR, dir_name, "images")
img_path = "/home/hyojiny1/yoon/data/CaltechPedestrians/extracted/images/"
img_ext = ".jpg"

#
def readAnno(filename):
    with open(filename) as json_config_file:
        js = json.load(json_config_file)
    return js


# frame starts from 1
if __name__ == "__main__":
    js = readAnno(anno_file)
    label_set = []
    for set_path in sorted(glob.glob(img_path + '/set*')):
        set_name = set_path.split("/")[-1]
        for video_path in sorted(glob.glob(set_path + '/V*')):
            video_name = video_path.split("/")[-1]
            unit = js[set_name][video_name]["frames"]
            for k, v in unit.items():
                fid = k
                label = unit[k][0]['lbl']

                # [l, t ,w ,h]

                pos = np.round(np.asarray(unit[k][0]['pos'], dtype=np.float32))
                area = pos[2]*pos[3]
                if (label not in label_set):
                    label_set.append(label)
                invisible = np.zeros((1,4),dtype=np.int32)
                # [x1 y1 x2 y2]
                coord = [pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]
                img = video_path + '/' + str(fid) + img_ext
                _str = unit[k][0]['str']
                lock = unit[k][0]['lock']
                if _str == 1:
                        print(_str)
                        """
                        im = cv2.imread(img, cv2.IMREAD_COLOR)
                        im = cv2.rectangle(im, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 1)
                        cv2.imshow(str(fid), im)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()
                        """
    print(label_set)