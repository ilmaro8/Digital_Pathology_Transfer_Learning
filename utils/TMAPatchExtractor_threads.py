from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import collections
import time
from skimage import io
import os
import pandas as pd
import threading

def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.mkdir(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path)

NUMBER_PATCHES = 30

PARTITION = 'train'
#PARTITION = 'valid'
#PARTITION = 'test'

LIST_IMAGES = '/home/niccolo/ExamodePipeline/MedIA_revision/csv_folder/partitions/missing_TMAs_'+PARTITION+'.csv'

DIR_NAME = '/mnt/nas4/datasets/ToReadme/prostate_TMAs/dataverse_files/TMA_images/'

IMAGE_dir = DIR_NAME + 'all/'
MASK_dir = DIR_NAME + 'Gleason_masks_train/'

list_files = csv_binary = pd.read_csv(LIST_IMAGES, sep=',', header=None, dtype=str).values.flatten()

OUTPUT_DIR = '/home/niccolo/ExamodePipeline/MedIA_revision/csv_folder/missing_TMA/'
create_dir(OUTPUT_DIR)

OUTPUT_DIR = OUTPUT_DIR+PARTITION+'/'
create_dir(OUTPUT_DIR)

THREAD_NUMBER = 3

patch_size = 750
resized_patch_size = 250
new_patch_size = 224
THRESHOLD = 0.3
size = 3100



filename_array = []
labels_array = []

lockTrain = threading.Lock()

def generate_glimpses_coords():
	
	x_boundary = size-patch_size-1
	y_boundary = size-patch_size-1

	#generate number
	x=np.random.randint(low=0,high=x_boundary)
	y=np.random.randint(low=0,high=y_boundary)
	x1 = x+patch_size
	y1 = y+patch_size

	return x,y,x1,y1

def check_patch(patch,threshold):
	b = False
	white = np.mean(patch)
	window_size = patch_size
	tot_pxl = window_size*window_size
	unique, counts = np.unique(patch, return_counts=True)
	elems = dict(zip(unique, counts))
	i = np.argmax(counts)

	class_p = unique[i]

	if (unique[i]!=4 and (counts[i]/tot_pxl)>threshold):
		b = True
	elif (white<180):
		b = True
		class_p = 0

	return b,class_p

def create_csv(nome_patch,labels,fname):
	File = {'filename':nome_patch,'gleason_pattern':labels}
	df = pd.DataFrame(File,columns=['filename','gleason_pattern'])
	#print(file_path)
	np.savetxt(fname, df.values, fmt='%s',delimiter=',')

def explore_list(list_files):
	print(list_files)

	for f in list_files:

		print(f)

		prefix = f[:4]

		img_filename = IMAGE_dir + f + '.jpg'
		mask_filename = MASK_dir + 'mask_' + f + '.png'

		threshold = THRESHOLD

			#open TMA image
		tma = Image.open(img_filename)
		tma_array = np.asarray(tma)

			#open mask file
		tma_mask = Image.open(mask_filename)
		tma_array_mask = np.asarray(tma_mask)

		
		i = 0
		cont = 0
		cont_threshold = 1000
		cont_max = 10000

		
		start_time = time.time()
		#print("analyzing " + str(filename))

		while(i<NUMBER_PATCHES and cont<cont_max):

			if (cont!=0 and cont%cont_threshold==0):
				threshold = threshold-0.05

			x_coords, y_coords, x1_coords, y1_coords = generate_glimpses_coords()
			mask_patch = tma_array_mask[y_coords:y1_coords,x_coords:x1_coords]
			flag, label = check_patch(mask_patch,threshold)

			if(flag):
				x = True
				tma_patch = tma_array[y_coords:y1_coords,x_coords:x1_coords,:]
				#tma_patch = img_augment_array[y_coords:y1_coords,x_coords:x1_coords,:]
					#resampling image
				new_im = Image.fromarray(tma_patch)

				new_im = new_im.resize((new_patch_size,new_patch_size))
				new_im = np.asarray(new_im)

				filename_output = OUTPUT_DIR+'/'+f+'_'+str(i)+'.jpg'
				io.imsave(filename_output, new_im)
				lockTrain.acquire()
				filename_array.append(filename_output)
				labels_array.append(label)
				lockTrain.release()

				#print("img done")

				i = i+1
				#cont = 0
			else:
				cont = cont+1


		elapsed_time = time.time() - start_time
		print("done in " + str(elapsed_time))

def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))


lists_files = list(chunker_list(list_files,THREAD_NUMBER))
print("lists_files " + str(type(lists_files)))

threads = []
for i in range(THREAD_NUMBER):
	t = threading.Thread(target=explore_list,args=[lists_files[i]])
	threads.append(t)

for t in threads:
	t.start()
for t in threads:
	t.join()

print("DONE")

create_csv(filename_array,labels_array,OUTPUT_DIR+PARTITION+'.csv')

