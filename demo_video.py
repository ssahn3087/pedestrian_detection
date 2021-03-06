import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.network import init_data
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.roi_data_layer.minibatch import process_img_by_lib

global video_file
global output_file
global fps
video_file = 'demo/Youtube_Pedestrians_720p.mp4'
output_file = 'demo/output.avi'
fps = 30
def test():
    import os
    imdb_name = 'CaltechPedestrians_test'
    imdb = get_imdb(imdb_name)
    cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
    model_dir = 'data/pretrained_model/'
    pre_model_name = 'CaltechPedestrians_train_10_vgg16_0.7_b3.h5'
    pretrained_model = model_dir + pre_model_name
    cfg_from_file(cfg_file)

    if 'vgg16' in pre_model_name.split('_'):
        detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
    elif 'res' in pre_model_name.split('_'):
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
                dets, scores, classes = detector.detect(frame, blob, thr=0.7, nms_thresh=0.3)
                frame = np.copy(frame)
                for i, det in enumerate(dets):
                    det = tuple(int(x) for x in det)
                    cv2.rectangle(frame, det[0:2], det[2:4], (255, 205, 51), 2)
                    # cv2.putText(frame, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), \
                    #             cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
                cv2.imshow('demo', frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            except IndexError as e:
                pass
            finally:
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
    test()
    # audio_target = video_file
    # video_target = output_file
    # file_name = '{}/{}.avi'.format(output_file.split('/')[0], 'sound_output')
    # start_AVrecording(video_target, audio_target)
    # stop_AVrecording(file_name)


