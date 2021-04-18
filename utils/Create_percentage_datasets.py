import os
import glob
import numpy as np
import argparse
argv = sys.argv[1:]

parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
args = parser.parse_args()

NUM_EXPERIMENT = args.N_EXP
N_EXP_str = str(NUM_EXPERIMENT)

center_train_prefix = ['ZT111','ZT199','ZT204']

TMA_arrays = []

csv_folder = #PATH_FOLDER
txt_dir = '/percentage_datasets_csvs/'

list_percentage = [0.8,0.6,0.4,0.2]
previous_percentage = 1

def extractor(list_files,new_len):
	new_list = np.random.choice(list_files,new_len,replace=False)
	return new_list

def create_csv(array,percentage):
	perc = int(percentage*100)
	csv_filename = txt_dir+'TMA_perc_'+str(perc)+'_'+str(NUM_EXPERIMENT)+'.txt'
	flattened = [val for sublist in array for val in sublist]
	np.savetxt(csv_filename, flattened, fmt='%s',delimiter=',')
	print(flattened)

print("create list_files")
for i in range(len(center_train_prefix)):
	list_files = glob.glob(path_TMA+center_train_prefix[i]+'**.jpg')
	TMA_arrays.append(list_files)
	print("list_files " + str(len(list_files)))

create_csv(TMA_arrays,previous_percentage)

for i in range(len(list_percentage)):
	print("percentage " + str(list_percentage[i]))
	for j in range(len(TMA_arrays)):
		num_elems = int(round((list_percentage[i]/previous_percentage)*len(TMA_arrays[j])))
		TMA_arrays[j] = extractor(TMA_arrays[j],num_elems)
		print("len " + str(len(TMA_arrays[j])))
	previous_percentage = list_percentage[i]
	create_csv(TMA_arrays,previous_percentage)

