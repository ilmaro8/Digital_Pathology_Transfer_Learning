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

argv = sys.argv[1:]

np.random.seed(0)

csv_test = '/home/niccolo/ExamodePipeline/gleason_challenge/patches/challenge_patches_strong_labels.csv'
#csv_test = '/home/niccolo/ExamodePipeline/gleason_challenge/patches_densely/challenge_patches_strong_labels.csv'

try:
    opts, args = getopt.getopt(argv,"hn:p:s:",["n_exp=","pre_trained=","subset="])
except getopt.GetoptError:
    print('Inference_TCGA_densely_keras.py -n <n_exp> -p <pre_trained> -s <subset>' )
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Inference_TCGA_densely_keras.py -n <n_exp> -p <pre_trained> -s <subset>')
        sys.exit()
    elif opt in ("-n", "-n_exp"):
        N_EXP_str = arg
    elif opt in ("-p", "-pre_trained"):
        EXPERIMENT = arg
    elif opt in ("-s", "-subset"):
        SUBSET = arg

try:
    print("PRE-TRAINED " + EXPERIMENT + " SUBSET " + SUBSET + " N_EXP " + N_EXP_str)
except:
    print("PRE-TRAINED " + EXPERIMENT + " N_EXP " + N_EXP_str)

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

#load_models
if (EXPERIMENT=='fully'):
    model_weights_dir = '/home/niccolo/ExamodePipeline/JHBI_review/model_weights/inference/exp_'+N_EXP_str+'/'+SUBSET+'_perc/'

model_path = model_weights_dir+'mobilenet_finetune_class.h5'
checkpoint_dir = model_weights_dir+'checkpoint_dir/'

create_dir(checkpoint_dir)

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

kappa_score_GPs_filename = checkpoint_dir+'kappa_patterns_patches_arvaniti.csv'
#kappa_score_GPs_filename = checkpoint_dir+'kappa_patterns_patches_challenge.csv'

kappas = [k_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_GPs_filename, df.values, fmt='%s',delimiter=',')

