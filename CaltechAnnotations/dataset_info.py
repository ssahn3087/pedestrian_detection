from faster_rcnn.datasets.factory import get_imdb
import PIL
import numpy as np
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

imdb_name = 'CaltechPedestrians'
global imdb
imdb = get_imdb(imdb_name)
roidb = imdb._roidb_handler()


def extract_hist_info(data, name, bins=7):
    statiscal_info = str(pd.Series(data).describe())
    print(statiscal_info)
    n, bins, patches = plt.hist(data, bins=bins)
    with open('./data/{}_{}.txt'.format(name, imdb.name), 'w') as f:
        f.write(name + '--' + imdb.name+'\n\n')
        f.write(statiscal_info)
        f.write('\n\ndata range     number\n')
        for i in range(len(n)):
            f.write('{:d}  ~  {:d}   :  {:d} \n' \
                    .format(int(bins[i]),int(bins[i+1]),int(n[i])))
    plt.xlabel(name)
    plt.ylabel('counts')
    plt.title('bbox histogram information')
    fig = plt.gcf()
    plt.show()
    fig.savefig('./data/{}_{}.png'.format(name, imdb.name))


num_images = imdb.num_images
box_info = np.zeros((0, 3))
for i in range(num_images):
    num_objs = roidb[i]['boxes'].shape[0]
    for j in range(num_objs):
        length_info = (roidb[i]['boxes'][j, -2:] - roidb[i]['boxes'][j, :2])
        area_info = roidb[i]['seg_areas'][j]
        box_info = np.vstack((box_info, np.hstack((length_info, area_info))))

width = box_info[:, 0]
height = box_info[:, 1]

# manually conditioned limit for general bbox information
indices = (np.where(width <= 100) and np.where(height <= 150))[0]


_width = box_info[indices, 0]
_height = box_info[indices, 1]
_area = box_info[indices, 2]
print('total {} data of {} analyzed'.format(len(indices),num_images))
extract_hist_info(_width, 'width')
extract_hist_info(_height, 'height')
extract_hist_info(_area, 'area')
mean_ratio = np.mean(_height/_width)
print('ratio mean width : height = {} : {}'.format(1,mean_ratio))

#sns.set()
#ax = sns.distplot(area, rug=True)

#sampled_indices = np.random.choice(num_images, 90000, replace=False)