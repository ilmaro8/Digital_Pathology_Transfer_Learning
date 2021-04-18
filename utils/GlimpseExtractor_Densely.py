import sys, os, getopt
import openslide
from PIL import Image
import numpy as np
import pandas as pd 
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import threading
import time
import collections
import cv2
import torch.utils.data
import albumentations as A
import time
import torch.nn.functional as F
import torch
from torchvision import transforms
from skimage import exposure

np.random.seed(0)

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv,"hd:",["dataset="])
except getopt.GetoptError:
    print('train.py -d <dataset> ')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('train.py -d <dataset> ')
        sys.exit()
    elif opt in ("-d", "-dataset"):
        DATASET_NAME = arg


SELECTED_LEVEL = 40
#SELECTED_LEVEL = 20
MASK_LEVEL = 1.25
MAGNIFICATION_RATIO = SELECTED_LEVEL/MASK_LEVEL
GLIMPSE_SIZE_SELECTED_LEVEL = 750
#GLIMPSE_SIZE_SELECTED_LEVEL = 375
GLIMPSE_SIZE_1x = int(GLIMPSE_SIZE_SELECTED_LEVEL/MAGNIFICATION_RATIO)
STRIDE_SIZE_1X = 0
TILE_SIZE_1X = GLIMPSE_SIZE_1x+STRIDE_SIZE_1X
PIXEL_THRESH = 0.7

print("SELECTED_MAGNIFICATION " + str(SELECTED_LEVEL) + "x")
print("MASK_MAGNIFICATION " + str(MASK_LEVEL) + "x")
print("GLIMPSE_SIZE_SELECTED_LEVEL " + str(GLIMPSE_SIZE_SELECTED_LEVEL))
print("GLIMPSE_SIZE_1.25x " + str(GLIMPSE_SIZE_1x))


def create_dir(models_path):
    if not os.path.isdir(models_path):
        try:
            os.mkdir(models_path)
        except OSError:
            print ("Creation of the directory %s failed" % models_path)
        else:
            print ("Successfully created the directory %s " % models_path)


LIST_FILE = #PATH_CSV_FILE 

PATH_INPUT_MASKS = #PATH_TISSUE_MASKS_GENERATED_WITH_HistoQC

PATH_OUTPUT = #PATH_WHERE_PATCHES_WILL_BE_STORED

GENERAL_TXT_PATH = PATH_OUTPUT+'csv_'+DATASET_NAME+'_densely.csv'

THREAD_NUMBER = 20
lockList = threading.Lock()
lockGeneralFile = threading.Lock()

def normalize_patterns(patterns):

	new_p = []

	for p in patterns:

		if (p==3):
			new_p.append(1)
		elif(p==4):
			new_p.append(2)
		elif(p==5):
			new_p.append(3)
		else:
			new_p.append(0)
	
	return new_p

#read right level
def right_level(file):
	try:
		magnification = file.properties['openslide.objective-power']
		#print("magnification found")
	except:
		magnification = 20.0
		#print("cannot find magnification")
	magnification = int(magnification) 
	curr_factor = int(magnification/SELECTED_LEVEL)
	index = 0
	for i in range(len(file.level_downsamples)):
		if (abs(file.level_downsamples[i]-curr_factor)<=1):
			index = i 
	#print("right level found")
	return index

#write incrementally to file (for every WSI)
def write_to_file(fname,arrays):
	path = os.path.normpath(fname)
	name_dir = path.split(os.sep)[-1]
	output_dir = PATH_OUTPUT+name_dir
	output_dir = os.path.abspath(PATH_OUTPUT+name_dir)
	File = {'filename':arrays[0],'level':arrays[1],'x_top':arrays[2],'y_top':arrays[3],'factor':arrays[4]}
	df = pd.DataFrame(File,columns=['filename','level','x_top','y_top','factor'])
	#file_path = output_dir+'/'+name_dir+'.csv'
	file_path = output_dir+'/'+name_dir+'_densely.csv'
	#print(file_path)
	np.savetxt(file_path, df.values, fmt='%s',delimiter=',')

def write_probs(fname,array_probs,array_labels):
	path = os.path.normpath(fname)
	name_dir = path.split(os.sep)[-1]
	output_dir = PATH_OUTPUT+name_dir
	output_dir = os.path.abspath(PATH_OUTPUT+name_dir)
	file_path_prob = output_dir+'/'+name_dir+'_probs.csv'
	file_path_labels = output_dir+'/'+name_dir+'_labels.csv'
	probs = np.asarray(array_probs)
	labels = np.array(array_labels)
	np.save(file_path_prob, probs)
	np.save(file_path_labels, labels)

	#check if dir exists and eventually it is created
def check_subdir_exist(fname):
		#create dir
	path = os.path.normpath(fname)
	name_dir = path.split(os.sep)[-1]
	output_dir = PATH_OUTPUT+name_dir
	if not os.path.exists(output_dir):
		print("create_output " + str(output_dir))
		os.makedirs(output_dir)
	return output_dir, name_dir

def write_general_csv(fname,arrays):
	File = {'filename':arrays[0],'primary_GG':arrays[1],'secondary_GG':arrays[2]}
	df = pd.DataFrame(File,columns=['filename','primary_GG','secondary_GG'])
	#print(file_path)
	np.savetxt(fname, df.values, fmt='%s',delimiter=',')

def create_output_imgs(img,fname):
	#save file
	new_patch_size = 224
	img = img.resize((new_patch_size,new_patch_size))
	img = np.asarray(img)
	#io.imsave(fname, img)
	print("file " + str(fname) + " saved")

#read incrementally to file
def read_file_tsv(fname):
	df = pd.read_csv(fname, sep="\t", header=None)
	values = df.values.flatten()
	return values

def read_file_csv(fname):
	df = pd.read_csv(fname, sep=',', header=None)
	print("list data")
	#return df['filename'],df['primary_GG'],df['secondary_GG']
	#return df[0].values,df[1].values,df[2].values
	return df[0].values,df[1].values,df[2].values
	#return df[0].values,df[3].values,df[4].values

def check_background(glimpse,threshold):
	b = False
	window_size = int(GLIMPSE_SIZE_SELECTED_LEVEL/MAGNIFICATION_RATIO)
	tot_pxl = window_size*window_size
	white_pxl = np.count_nonzero(glimpse)
	score = white_pxl/tot_pxl
	if (score>=threshold):
		b=True
	return b

def whitish_img(img):
	THRESHOLD_WHITE = 200
	b = True
	if (np.mean(img) > THRESHOLD_WHITE):
		b = False
	return b

#estrae glimpse e salva metadati relativi al glimpse
def analyze_file(filename,gleason_score,patch_score):
	global filename_list_general, gleason_scores_general, patch_scores_general
	global probs_array
	global labels_array

	patches = []

	new_patch_size = 224
		#load file
	#file = openslide.OpenSlide(filename)
	file = openslide.open_slide(filename)
	level = right_level(file)
		#load mask
	path = os.path.normpath(filename)
	fname = path.split(os.sep)[-1]
		#check if exists
	fname_mask = PATH_INPUT_MASKS+fname+'/'+fname+'_mask_use.png' 

	array_dict = []

	print(fname)

	if os.path.isfile(fname_mask):

			#creates directory
		output_dir, name_file = check_subdir_exist(filename)
			#create CSV file structure (local)
		filename_list = []
		level_list = []
		x_list = []
		y_list = []
		factor_list = []

		img = Image.open(fname_mask)
		img = np.asarray(img)

		tile_x = int(img.shape[1]/TILE_SIZE_1X)
		tile_y = int(img.shape[0]/TILE_SIZE_1X)
		n_image = 0
		threshold = PIXEL_THRESH

		for i in range(tile_y):
			for j in range(tile_x):
				y_ini = int(TILE_SIZE_1X*i)
				x_ini = int(TILE_SIZE_1X*j)

				glimpse = img[y_ini:y_ini+GLIMPSE_SIZE_1x,x_ini:x_ini+GLIMPSE_SIZE_1x]

				if(check_background(glimpse,threshold)):
					#print("glimpse accepted ")
						#change to magnification 10x
					fname = os.path.abspath(output_dir+'/'+name_file+'_'+str(n_image)+'.png')
					x_coords = x_ini*MAGNIFICATION_RATIO
					y_coords = y_ini*MAGNIFICATION_RATIO
						#change to magnification 40x
					x_coords_0 = int(x_coords*file.level_downsamples[level])
					y_coords_0 = int(y_coords*file.level_downsamples[level])

					file_40x = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_SIZE_SELECTED_LEVEL,GLIMPSE_SIZE_SELECTED_LEVEL))
					file_40x = file_40x.convert("RGB")
					
					new_patch_size = 224
					save_im = file_40x.resize((new_patch_size,new_patch_size))

					#if (whitish_img(save_im) and exposure.is_low_contrast(save_im)==False):
					if (exposure.is_low_contrast(save_im)==False):
						save_im = np.asarray(save_im)
						io.imsave(fname, save_im)
						
						#add to arrays (local)
						filename_list.append(fname)
						level_list.append(level)
						x_list.append(x_coords_0)
						y_list.append(y_coords_0)
						factor_list.append(file.level_downsamples[level])
						n_image = n_image+1
					#save the image
					#create_output_imgs(file_10x,fname)
		
			#add to general arrays
		lockGeneralFile.acquire()
		filename_list_general.append(output_dir)
		gleason_scores_general.append(gleason_score)
		patch_scores_general.append(patch_score)
		
		print("len filename " + str(len(filename_list_general)))
		try:
			write_probs(filename,probs_array,(gleason_score,patch_score))
		except:
			pass
		lockGeneralFile.release()
		write_to_file(filename,[filename_list,level_list,x_list,y_list,factor_list])
	
	else:

		print("no mask")

def explore_list(list_dirs,primary_gleason_patterns,secondary_gleason_patterns):
	global list_dicts, n
	
	
	#print(threadname + str(" started"))


	for i in range(len(list_dirs)):
		analyze_file(list_dirs[i],primary_gleason_patterns[i],secondary_gleason_patterns[i])
	#print(threadname + str(" finished"))



#list of lists fname-bool
def create_list_dicts(filenames,gs,ps):
	n_list = []
	for (f,g,p)in zip(filenames,gs,ps):
		dic = {"filename":f,"primary_GG":g,"secondary_GG":p,"state":False}
		n_list.append(dic)
	return n_list

def main():
	#create output dir if not exists
	start_time = time.time()
	global list_dicts, n, filename_list_general, gleason_scores_general, patch_scores_general


		#create CSV file structure (global)
	filename_list_general = []
	gleason_scores_general = []
	patch_scores_general = []

	n = 0
		#create dir output
	if not os.path.exists(PATH_OUTPUT):
		print("create_output " + str(PATH_OUTPUT))
		os.makedirs(PATH_OUTPUT)

	list_dirs, primary_gleason_patterns, secondary_gleason_patterns = read_file_csv(LIST_FILE)
	
	#primary_gleason_patterns = normalize_patterns(primary_gleason_patterns)
	#secondary_gleason_patterns = normalize_patterns(secondary_gleason_patterns)

	def chunker_list(seq, size):
		return (seq[i::size] for i in range(size))

	list_dirs = list(chunker_list(list_dirs,THREAD_NUMBER))
	primary_gleason_patterns = list(chunker_list(primary_gleason_patterns,THREAD_NUMBER))
	secondary_gleason_patterns = list(chunker_list(secondary_gleason_patterns,THREAD_NUMBER))

	threads = []
	for i in range(THREAD_NUMBER):
		t = threading.Thread(target=explore_list,args=(list_dirs[i],primary_gleason_patterns[i],secondary_gleason_patterns[i]))
		threads.append(t)

	for t in threads:
		t.start()
		#time.sleep(60)

	for t in threads:
		t.join()

	write_general_csv(GENERAL_TXT_PATH,[filename_list_general,gleason_scores_general,patch_scores_general])
	
		#prepare data
	
	elapsed_time = time.time() - start_time
	print("elapsed time " + str(elapsed_time))
if __name__ == "__main__":
	main()

