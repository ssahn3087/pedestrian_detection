import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import numpy.random as npr
from faster_rcnn import network
from faster_rcnn.network import init_data
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
import cv2
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

def draw_graph(fig, name):
    plt.title('Embedded Anchors')
    fig.savefig(name+'.jpg')
# hyper-parameters
# ------------

imdb_name = 'CaltechPedestrians_test_triplet'

cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
model_dir = 'data/pretrained_model/'
output_dir = 'models/saved_model3'
pre_model_name = 'CaltechPedestrians_train_triplet_1_vgg16_cls_0.7_b3.h5'
pretrained_model = model_dir + pre_model_name

_DEBUG = False
BG_SHOW = True
id_limit = 20 if BG_SHOW else 50
# load config
cfg_from_file(cfg_file)
# load data
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)

detector = FasterRCNN_VGG(classes=imdb.classes, debug=_DEBUG)
network.load_net(pretrained_model, detector)

blob = init_data(is_cuda=True)

detector.cuda()
detector.eval()
name_blocks = pre_model_name.split('_')
batch_size = imdb.num_triplet_test_images
test_num = len(roidb)
blob = init_data(is_cuda=True)
features = []
bg_features = []
ids = []
print('Extracting features...')
t = Timer()
t.tic()
for i in range(test_num):
    image = cv2.imread(roidb[i]['image'])
    gt_boxes = roidb[i]['boxes'].astype(np.float32)
    relu = True if 'relu' in name_blocks else False
    features.append(detector.extract_feature_vector(image, blob, gt_boxes, relu=relu).data.cpu().numpy())
    ids.append(roidb[i]['ids'][0])
    if BG_SHOW:
        bg_features.append(detector.extract_background_features(image, blob, gt_boxes, relu=relu).data.cpu().numpy())
    if len(set(ids)) > id_limit:
        break

print('{:3.2f}s feature extraction finished !'.format(t.toc(average=False)))
features = np.asarray(features, dtype=np.float32)
learning_rate = 100
model = TSNE(learning_rate=learning_rate)
labels = np.array(ids) % 4
fig, ax = plt.subplots()
font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 7,
        }
# Positive Anchors
pos_data = model.fit_transform(features)
xs = pos_data[:, 0]
ys = pos_data[:, 1]
ax.scatter(xs, ys, marker='+', c=labels)
for i, id in enumerate(ids):
    ax.text(xs[i], ys[i], str(id), fontdict=font)
# Negative(Background) Anchors
if BG_SHOW:
    neg_data = model.fit_transform(bg_features)
    xs = neg_data[:, 0]
    ys = neg_data[:, 1]
    ax.scatter(xs, ys, marker='x', c='r')
    font['color'] = 'black'
    for i, id in enumerate(ids):
        ax.text(xs[i], ys[i], 'bg_' + str(id), fontdict=font)

plt.show()
pre_model_name = pre_model_name.replace('.h5', '_'+str(learning_rate))
pre_model_name += '_bg' if BG_SHOW else ''
draw_graph(fig, pre_model_name)