import json
import os
import glob
import cv2



def save_img(fn, i, frame):
	return cv2.imwrite('{}/{}.jpg'.format(fn, i), frame)


def make_dir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
def load_json_file(annotations_path):
	with open(annotations_path+"/annotations.json") as json_file:
    		data = json.load(json_file)
	return data
# use: python extract.images.py where "annotations, code, seq_files" directory exists
# annotation file must exist before executing
annotations_path = os.path.join(os.getcwd(), 'extracted/annotations')
out_dir = os.path.join(os.getcwd(), 'extracted/images')
make_dir(out_dir)
input_dir = os.path.join(os.getcwd(), 'seq_files')
cnt = 1

anno_file = load_json_file(annotations_path)
f = open(out_dir+"/ref.txt",'w')
for dname in sorted(glob.glob(input_dir+'/set*')):
	dir_each_set = out_dir+'/'+ dname.split('/')[-1]
	make_dir(dir_each_set)
	for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
		dir_each_seq = dir_each_set+'/'+(fn.replace(".seq","")).split('/')[-1]

		arg = dir_each_seq.split('/')
		#arg[-2] = 'set**' / arg[-1] = 'V***'
		set_name = arg[-2]
		video_name = arg[-1]

		make_dir(dir_each_seq)
		cap = cv2.VideoCapture(fn)

		i = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			if str(i) in anno_file[set_name][video_name]['frames']:
				if(save_img(dir_each_seq, i, frame)):
					arg = dir_each_seq.split('/')
					f.write(set_name+"/"+video_name+"/"+str(i)+"/"+str(cnt)+"\n")
					cnt = cnt + 1
			i = i + 1	
		print(fn)
f.close()
print(cnt,'images saved')


"""
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for dname in sorted(glob.glob('data/set*')):
    for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
        cap = cv.VideoCapture(fn)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            save_img(dname, fn, i, frame)
            i += 1
	print(fn)
"""
