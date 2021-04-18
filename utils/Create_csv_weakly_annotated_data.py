import pandas as pd
import numpy as np
import os
import argparse
argv = sys.argv[1:]

parser.add_argument('-d', '--DATASET', help='partition',type=str, default='train')
args = parser.parse_args()

dataset_analyzed = args.DATASET


path_images = #PATH_DATA 
path_csv = #PATH_WITH_CSV_WITH_WSI_IDs

csv_train = path_csv + 'csv_train.csv'
csv_test = path_csv + 'csv_test.csv'
csv_valid = path_csv + 'csv_valid.csv'

path_data_train = path_images + 'train/'
path_data_valid = path_images + 'valid/'
path_data_test = path_images + 'test/'


filename = []
primary = []
secondary = []
scores = []

if (dataset_analyzed == 'train'):
    data_csv = pd.read_csv(csv_train,header=None).values
    path_data = path_data_train
elif (dataset_analyzed == 'valid'):
    data_csv = pd.read_csv(csv_valid,header=None).values
    path_data = path_data_valid
else:
    data_csv = pd.read_csv(csv_test,header=None).values
    path_data = path_data_test

list_dirs = os.listdir(path_data)

#print(list_dirs)
#print(data_csv)

def gleason_score(a,b):
	gs = a+b-2
	if (gs==1 and a==1):
		gs = 1
	elif (gs==1 and a==2):
		gs = 2
	elif (gs==2):
		gs = 3
	elif (gs>2):
		gs = 4
	return gs

for d in list_dirs:
	i = 0
	b = False
	while (i<len(data_csv) and b==False):
		path = data_csv[i,0]
		x = os.path.split(path)[1]

		source = d[:12]
		dest = x[:12]
		if (source==dest):
			p = data_csv[i,1]
			s = data_csv[i,2]
			gs = gleason_score(p,s)
			b = True
			list_files = os.listdir(path)
			for f in list_files:
				if(f[-3:]=='png'):
					file = path+'/'+f

					filename.append(file)
					primary.append(p)
					secondary.append(s)
					scores.append(gs)
		i = i+1

output_dir = path_csv + dataset_analyzed+'_finetuning.csv'

File = {'filename':filename,'primary_GG':primary,'secondary_GG':secondary,'gleason_score':scores}
df = pd.DataFrame(File,columns=['filename','primary_GG','secondary_GG','gleason_score'])
#print(file_path)
np.savetxt(output_dir, df.values, fmt='%s',delimiter=',')