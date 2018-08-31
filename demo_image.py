import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.network import init_data
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.datasets.factory import get_imdb


def test():
    import os
    img_file = 'demo/004545.jpg'
    image = cv2.imread(img_file)

    #imdb_name = 'CaltechPedestrians_train'
    imdb_name = 'voc_2007_trainval'
    imdb = get_imdb(imdb_name)
    cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
    model_dir = 'data/pretrained_model/'
    pre_model_name = 'resnet50_pedestrians_230000_0.7_b1.h5'
    pretrained_model = model_dir + pre_model_name
    cfg_from_file(cfg_file)
    print(imdb.classes)
    if 'vgg16' in pre_model_name.split('_'):
        detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
    elif 'resnet50' in pre_model_name.split('_'):
        detector = FasterRCNN_RES(classes=imdb.classes, debug=False)
    else:
        detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
    network.load_net(pretrained_model, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    blob = init_data(is_cuda=True)


    t = Timer()
    t.tic()
    dets, scores, classes = detector.detect(image, blob, thr=0.7, nms_thresh=0.3)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15),\
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)
    cv2.imshow('demo', im2show)
    cv2.waitKey(0)


if __name__ == '__main__':
    test()
