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
import os
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

CSV_FOLDER = #path_csv
csv_test = CSV_FOLDER + 'csv_training_cores.csv'

THRESHOLD = 30

test_dir = #DIR_WITH_IMAGES

try:
    print("PRE-TRAINED " + EXPERIMENT + " SUBSET " + SUBSET + " N_EXP " + N_EXP_str)
except:
    print("PRE-TRAINED " + EXPERIMENT + " N_EXP " + N_EXP_str)

#load testing data
data_test = pd.read_csv(csv_test,header=None).values

data_test_paths = data_test[:,0]
data_test_labels = data_test[:,1:]
data_test_labels = data_test_labels.astype('int64')
print(data_test.shape)
print(data_test_paths.shape)
print(data_test_labels.shape)

array_probs = []

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
        if(check_file(file)):
            img = Image.open(file)
            img = np.asarray(img)
            dataset.append(img)
    dataset = np.array(dataset)
    return dataset

def find_first_two(array):
    x = np.copy(array)
    max_1 = np.argmax(x)
    max_val1 = x[max_1]
    x[max_1]=-1
    max_2 = np.argmax(x)
    max_val2 = x[max_2]
    
    if(max_1==0 or max_2==0):
        max_1 = max(max_1,max_2)
        max_2 = max(max_1,max_2)
    
    if (max_val1>(2*max_val2)):
        max_2 = max_1
    
    return max_1,max_2

def majority_voting(array):
    majority = [0,0,0,0]
    for i in range(array.shape[0]):
        #print(prob)
        idx = np.argmax(array[i])
        majority[idx] = majority[idx]+1
    #majority[0]=0
    pgp, sgp = find_first_two(majority)
    return pgp, sgp, majority

def load_and_evaluate(list_f,elems):

    array_probs = []

    imgs = load_data(list_f)
    imgs = preprocess_input(imgs)
    probs = model.predict(imgs)
    array_probs.append(probs)

    array_probs = np.reshape(array_probs,(elems,4))
    #array_probs = np.squeeze(array_probs)

    #majority voting
    pgp,sgp, histogram = majority_voting(array_probs)
    y_preds.append([pgp,sgp])


def assign_group(a, b, survival_groups=False):
    # if both cancer and benign tissue are predicted
    # ignore benign tissue for reporting, as pathologists do
    if (a > 0) and (b == 0):
        b = a
    if (b > 0) and (a == 0):
        a = b

    # get the actual Gleason pattern (range 3-5)
    gs = 0
    if (a+b)==2:
        gs=1
    elif (a==1 and b==2):
        gs=2
    elif (a==2 and b==1):
        gs=3
    elif (a+b)==4:
        gs=4
    elif (a+b)>4:
        gs=5
    return gs

def pathologist_evaluation(array):
    new_evaluation = []
    for i in range(len(array)):
        gg = assign_group(array[i,0],array[i,1],True)
        new_evaluation.append(gg)
    new_evaluation = np.array(new_evaluation)
    return new_evaluation

def gleason_score(primary,secondary):
    
    array = []
    for i in range(len(primary)):

        a = primary[i]
        b = secondary[i]

        if (a > 0) and (b == 0):
            b = a
        if (b > 0) and (a == 0):
            a = b

        gs = a+b
        
        if (a==1 and b==1):
            gs = 1
        elif (a==2 and b==1):
            gs = 3
        elif (a==1 and b==2):
            gs = 2
        elif (gs==4):
            gs = 4
        elif (gs>4):
            gs = 5
        array.append(gs)
              
    return array

def predict_metrics(y_pred,y_true,metric):
    if(metric=='primary'):
        #primary gleason pattern
        y_true = y_true[:,0]
        y_pred = y_pred[:,0]
        k_score_primary = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        
        print("k_score_primary " + str(k_score_primary))
        
        return k_score_primary
        
    elif(metric=='secondary'):
        #secondary gleason pattern
        y_true = y_true[:,1]
        y_pred = y_pred[:,1]
        k_score_secondary = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        
        print("k_score_secondary " + str(k_score_secondary))
        
        return k_score_secondary
        
    else:
        #gleason score
        #y_true = y_true[:,0]+y_true[:,1]
        #y_pred = y_pred[:,0]+y_pred[:,1]

        y_true = gleason_score(y_true[:,0],y_true[:,1])
        y_pred = gleason_score(y_pred[:,0],y_pred[:,1])
        
        #y_true = pathologist_evaluation(y_true)
        #y_pred = pathologist_evaluation(y_pred)

        #print(y_pred)
        k_score_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
                
        print("k_score_score " + str(k_score_score))
        confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        print("confusion_matrix ")
        print(str(confusion_matrix))
        return k_score_score

y_preds = []

for p in data_test_paths:
    d = os.path.split(p)[1][:-4]
    directory = test_dir+d
    #csv_file = directory+'/'+d+'_densely.csv'
    csv_file = directory+'/'+d+'_labels_densely.csv'

    local_csv = pd.read_csv(csv_file,header=None).values[:,0]

    #print(local_csv.shape,len(local_csv))

    local_csv = local_csv[:THRESHOLD]

    #print(THRESHOLD,local_csv.shape,len(local_csv))

    load_and_evaluate(local_csv,len(local_csv))

#METRICS

y_preds = np.array(y_preds)
y_true = data_test_labels

y_preds = y_preds.astype('int64')
"""
for i in range(len(y_preds)):
    print(y_preds[i],y_true[i])    
"""

kappa_score_primary = predict_metrics(y_preds,y_true,'primary')
kappa_score_secondary = predict_metrics(y_preds,y_true,'secondary')
kappa_score_score = predict_metrics(y_preds,y_true,'score')

kappa_score_best_PGP_filename = checkpoint_dir+'kappa_score_PGP_cores_challenge.csv'
kappa_score_best_SGP_filename = checkpoint_dir+'kappa_score_SGP_cores_challenge.csv'
kappa_score_best_GS_filename = checkpoint_dir+'kappa_score_GS_cores_challenge.csv'

kappas = [kappa_score_primary]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_PGP_filename, df.values, fmt='%s',delimiter=',')

kappas = [kappa_score_secondary]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_SGP_filename, df.values, fmt='%s',delimiter=',')

kappas = [kappa_score_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_GS_filename, df.values, fmt='%s',delimiter=',')