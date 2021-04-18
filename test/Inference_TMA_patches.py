import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import logging
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import warnings
warnings.filterwarnings("ignore")

import keras
from PIL import Image
from keras.applications.mobilenet import MobileNet,preprocess_input
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import regularizers
import json
import copy
from sklearn import metrics
import glob
import random
import collections
import time
from keras import regularizers
import sys, getopt

import argparse
argv = sys.argv[1:]

parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-e', '--EXPERIMENT', help='kind of supervision: fully, weak, finetuning',type=str, default='fully')
parser.add_argument('-p', '--PERCENTAGE', help='percentange to use (20,40,60,80,100)',type=int, default=100)
args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
PERCENTAGE_DATA = args.PERCENTAGE
PERCENTAGE_DATA_str = str(PERCENTAGE_DATA)
TYPE_OF_SUPERIVISION = args.EXPERIMENT

np.random.seed(0)

csv_folder = #PATH_CSV

csv_test = csv_folder + 'TMAZ_test_partition.csv'

models_dir = #PATH_MODELS

#load_models
if (EXPERIMENT=='fully'):
    models_dir = models_dir + 'fully_supervision/'
    models_dir = models_dir + 'perc_' + PERCENTAGE_DATA_str + '/'
    models_dir = models_dir + 'N_EXP_' + N_EXP_str + '/'
    model_path = models_dir+'mobilenet_fully_supervision.h5'
elif (EXPERIMENT=='weak'):
    models_dir = models_dir + 'weak_supervision/'
    models_dir = models_dir + '/exp_'+N_EXP+'/'
    model_path = models_dir+'mobilenet_weak_supervision.h5'
elif (EXPERIMENT=='finetune'):
    models_dir = models_dir + 'finetuning/'
    models_dir = models_dir + 'perc_' + PERCENTAGE_DATA_str + '/'
    models_dir = models_dir + '/exp_'+N_EXP+'/'
    model_path = models_dir+'mobilenet_finetune_WSI_class_strong.h5'


checkpoint_dir = models_dir+'checkpoint_dir/'

create_dir(checkpoint_dir)

#load testing data
data_test = pd.read_csv(csv_test,header=None).values

data_test_paths = data_test[:,0]
data_test_labels = data_test[:,1]
print(data_test.shape)
print(data_test_paths.shape)
print(data_test_labels.shape)


#def models
pre_trained_model = MobileNet(weights='imagenet')
#config 6837555: mobilenet, 20 patches per TMA
pre_trained_model.layers.pop()
#fc1 = Dense(512,activity_regularizer=regularizers.l2(0.01))(pre_trained_model.layers[-1].output)
fc1 = Dense(512,activity_regularizer=regularizers.l2(0.01))(pre_trained_model.layers[-1].output)
d1 = Dropout(0.2)(fc1)
fc2 = Dense(4, activation='softmax',activity_regularizer=regularizers.l2(0.01))(d1)

model = Model(inputs=pre_trained_model.input, outputs=fc2)

def create_dir(directory):
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory %s failed" % directory)



try:
    model.load_weights(model_path)
    print("WEIGHTS LOADED")
except:
    print("weights not found in: " + str(model_path))
    pass

def check_file(f):
    suffix = '.png' 
    b = f.endswith(suffix)
    return b

def load_data(list_f):
    dataset = []
    #print(list_f)
    for i in range(len(list_f)):
        #file = directory+'/'+list_f[i]
        file = list_f[i]
        img = Image.open(file)
        img = np.asarray(img)
        dataset.append(img)
    dataset = np.array(dataset)
    return dataset

y_true = np.array(data_test_labels)

imgs = load_data(data_test_paths)
imgs = preprocess_input(imgs)
y_preds = model.predict(imgs)

y_preds = np.argmax(y_preds,axis=1)
print(y_preds.shape,y_true.shape)

#METRICS
y_preds = y_preds.tolist()
y_true = y_true.tolist()

k_score = metrics.cohen_kappa_score(y_preds,y_true, weights='quadratic')
print("kappa " + str(k_score))

kappa_score_GPs_filename = checkpoint_dir+'kappa_score.csv'
#kappa_score_GPs_filename = checkpoint_dir+'kappa_patterns_patches_challenge.csv'

kappas = [k_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_GPs_filename, df.values, fmt='%s',delimiter=',')

