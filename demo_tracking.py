import cv2
import numpy as np
import torch
from faster_rcnn import network
from faster_rcnn.network import init_data
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.utils.cython_bbox import bbox_overlaps

global video_file
global output_file
global fps
video_file = 'demo/Gang burgle 50 firearms from gun shop in 2 minutes, Houston.mp4'
output_file = 'demo/output.avi'
fps = 30
tps = 6

def track():
    def id_track(dataset, features):
        from collections import Counter
        def dist(f1, f2):
            score = (torch.sqrt((f1 - f2) ** 2)).sum(0).data.cpu().numpy()[0]
            return score

        id_list = []
        id_count = {'f' + str(i): [] for i in range(len(features))}
        for dataframe in dataset:
            for i, f in enumerate(features):
                init_val = 1e15
                for data in dataframe:
                    score = dist(f, data['feature'])
                    if score < init_val:
                        init_val = score
                        id = data['id']
                id_count['f' + str(i)].append(id)
        for list in id_count.values():
            c1 = Counter(list)
            most_id = c1.most_common(1)[0][0]
            id_list.append(most_id)
        return id_list
    import os
    imdb_name = 'CaltechPedestrians_test'
    imdb = get_imdb(imdb_name)
    cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
    model_dir = 'data/pretrained_model/'
    pre_model_name = 'CaltechPedestrians_train_2_vgg16_0.7_b3.h5'
    pretrained_model = model_dir + pre_model_name
    cfg_from_file(cfg_file)
    name_blocks = pre_model_name.split('_')
    if 'vgg16' in name_blocks:
        detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
    elif 'resnet50' in name_blocks:
        detector = FasterRCNN_RES(classes=imdb.classes, debug=False)
    else:
        detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
    relu = True if 'relu' in name_blocks else False
    network.load_net(pretrained_model, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    blob = init_data(is_cuda=True)

    t = Timer()
    t.tic()
    cap = cv2.VideoCapture(video_file)
    init = True
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            p = Timer()
            p.tic()

            if init:
                cnt = 1
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_file, fourcc, fps, (frame.shape[1], frame.shape[0]))
                init = False
            try:
                # detect
                tid = (cnt-1) % tps
                dets, scores, classes = detector.detect(frame, blob, thr=0.7, nms_thresh=0.3)
                frame = np.copy(frame)
                # feature extraction
                features = []
                for i, det in enumerate(dets):
                    gt_box = det[np.newaxis,:]
                    features.append(detector.extract_feature_vector(frame, blob, gt_box, relu=relu))
                    det = tuple(int(x) for x in det)
                    cv2.rectangle(frame, det[0:2], det[2:4], (255, 205, 51), 2)
                dataframe = []
                if tid == 0:
                    dataset = []
                    for i, f in enumerate(features):
                        data = {}
                        data['id'] = i
                        data['feature'] = f
                        dataframe.append(data)
                    dataset.append(dataframe)
                    anchors = dets
                elif tid > 0 and tid < tps-1:
                    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float) \
                                             , np.ascontiguousarray(dets, dtype=np.float))
                    # max : K max overlaps score about N dets
                    overlaps = np.multiply(overlaps, overlaps > 0.7)
                    max_arg = overlaps.argmax(axis=0)
                    for i, arg in enumerate(max_arg):
                        if arg >= len(features):
                            continue
                        data = {}
                        data['id'] = arg
                        data['feature'] = features[arg]
                        dataframe.append(data)
                    dataset.append(dataframe)
                    anchors = dets
                else:
                    id_list = id_track(dataset, features)
                    for i, id in enumerate(id_list):
                        det = tuple(int(x)-2 for x in dets[i])
                        cv2.putText(frame, 'id: ' + str(id), det[0:2], cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255))
                    # cv2.imshow('demo', frame)
                    # cv2.waitKey(1000)
                    # cv2.destroyAllWindows()
            except:
                pass
            finally:
                if cnt % 10 == 0:
                    print(cnt,'-frame : {:.3f}s'.format(p.toc()))
                cnt += 1
                out.write(frame)
        else:
            break
    runtime = t.toc()
    print('{} frames  /  total spend: {}s  /  {:2.1f} fps'.format(cnt, int(runtime), cnt/runtime))
    cap.release()
    out.release()

if __name__ == '__main__':
    track()


