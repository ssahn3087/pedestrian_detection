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


input_dir = os.path.join(os.getcwd(), 'seq_files')

train_set = ['set00','set01','set02','set03','set04','set05']
test_set = ['set06','set07','set08','set09','set10']



anno_file = load_json_file(annotations_path)
init1 = True
init2 = True
for dname in sorted(glob.glob(input_dir+'/set*')):
	set_name = dname.split('/')[-1]
	if set_name in train_set:
		cnt = 1
		out_dir = os.path.join(os.getcwd(), 'extracted/images', 'train')
		make_dir(out_dir)
		dir_each_set = os.path.join(out_dir,set_name)
		make_dir(dir_each_set)
		if init1 :
			f1 = open(out_dir + "/train.txt",'w')
			init1 = False
		else:
			f1 = open(out_dir + "/train.txt",'a')
		for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
			i = 0
			dir_each_seq = os.path.join(dir_each_set,(fn.replace(".seq","")).split('/')[-1])
			arg = dir_each_seq.split('/')
			#arg[-2] = 'set**' / arg[-1] = 'V***'
			video_name = arg[-1]
			make_dir(dir_each_seq)
			cap = cv2.VideoCapture(fn)
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				if str(i) in anno_file[set_name][video_name]['frames']:
					if(save_img(dir_each_seq, i, frame)):
						arg = dir_each_seq.split('/')
						f1.write(set_name+"/"+video_name+"/"+str(i)+"/"+str(cnt)+"\n")
						cnt = cnt + 1
				i = i + 1	
			print(fn)
	elif set_name in test_set:
		cnt = 1
		out_dir = os.path.join(os.getcwd(), 'extracted/images', 'test')
		make_dir(out_dir)
		dir_each_set = os.path.join(out_dir,set_name)
		make_dir(dir_each_set)
		if init2 :
			f2 = open(out_dir + "/test.txt",'w')
			init2 = False
		else:
			f2 = open(out_dir + "/test.txt",'a')
		for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
			i = 0
			dir_each_seq = os.path.join(dir_each_set,(fn.replace(".seq","")).split('/')[-1])
			arg = dir_each_seq.split('/')
			#arg[-2] = 'set**' / arg[-1] = 'V***'
			video_name = arg[-1]
			make_dir(dir_each_seq)
			cap = cv2.VideoCapture(fn)
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				if str(i) in anno_file[set_name][video_name]['frames']:
					if(save_img(dir_each_seq, i, frame)):
						arg = dir_each_seq.split('/')
						f2.write(set_name+"/"+video_name+"/"+str(i)+"/"+str(cnt)+"\n")
						cnt = cnt + 1
				i = i + 1	
			print(fn)
f1.close()
f2.close()
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
