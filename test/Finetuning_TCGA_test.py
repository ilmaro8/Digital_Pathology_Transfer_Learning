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
from tqdm import tqdm

np.random.seed(0)

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#PATHS
    #if 100%
csv_test = '/home/niccolo/ExamodePipeline/WSI_patches/3000/test_3000/csv_test_3000.csv'
models_path = '/home/niccolo/ExamodePipeline/JHBI_review/model_weights/finetuning_weak/weak_finetuning/'
#models_path = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/model_weights/finetuning/strong_finetuning/'
#models_path = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/model_weights/finetuning/strong_finetuning_20/'

test_dir = '/home/niccolo/ExamodePipeline/WSI_patches/3000/test_3000/'

print("model path " + str(models_path))

#load testing data
data_test = pd.read_csv(csv_test,header=None).values

data_test_paths = data_test[:,0]
data_test_labels = data_test[:,1:]
data_test_labels = data_test_labels.astype('int64')
print(data_test.shape)
print(data_test_paths.shape)
print(data_test_labels.shape)

def check_file(f):
    suffix = '.png' 
    b = f.endswith(suffix)
    return b

array_probs = []

#load_models
models = []
for i in tqdm(range(10)):
    model_filename = models_path+'exp_'+str(i)+'/mobilenet_finetune_WSI_class_weak.h5'
    #model_filename = models_path+'exp_'+str(i)+'/mobilenet_finetune_WSI_class_strong.h5'

    pre_trained_model = MobileNet(weights='imagenet')
    
    #config 6837555: mobilenet, 20 patches per TMA
    pre_trained_model.layers.pop()
    #fc1 = Dense(512,activity_regularizer=regularizers.l2(0.01))(pre_trained_model.layers[-1].output)
    fc1 = Dense(512)(pre_trained_model.layers[-1].output)
    d1 = Dropout(0.2)(fc1)

    #primary classifier
    x1 = Dense(3, activation="softmax",activity_regularizer=regularizers.l2(0.01), name='output_1')(d1)

    #secondary classifier
    x2 = Dense(3, activation="softmax",activity_regularizer=regularizers.l2(0.01), name='output_2')(d1)

    model = Model(inputs=pre_trained_model.input, outputs=[x1, x2])

    opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-3 / 100)
    model.compile(loss={'output_1': 'categorical_crossentropy', 'output_2': 'categorical_crossentropy'},optimizer=opt, metrics=['acc'])
    #model.summary()

    try:
        model.load_weights(model_filename)
        print("WEIGHTS LOADED")
        models.append(model)
        array_probs.append([])
    except:
        print("weights not found in: " + str(model_filename))
        pass

def print_prediction(filename, y_true, y_pred):
    print("filename:  " + str(filename))
    print("y_true: " + str(y_true))
    print("y_pred: " + str(y_pred))
"""
def load_data(list_f):
    dataset = []
    for i in tqdm(range(len(list_f))):
        file = directory+'/'+list_f[i]
        if(check_file(file)):
            img = Image.open(file)
            img = np.asarray(img)
            dataset.append(img)
    dataset = np.array(dataset)
    return dataset
"""

def load_data(list_f):
    dataset = []
    for i in tqdm(range(len(list_f))):
        if(check_file(list_f[i])):
            img = Image.open(list_f[i])
            img = np.asarray(img)
            dataset.append(img)
    dataset = np.array(dataset)
    return dataset

cont = 0
for p in data_test_paths:
    print(cont)
    d = os.path.split(p)[1]
    directory = test_dir+d
    csv_file = directory+'/'+d+'_3000_sorted_br_patches.csv'

    files = pd.read_csv(csv_file,header=None).values[:1000,0]
    #files = os.listdir(directory)
    imgs = load_data(files)
    imgs = preprocess_input(imgs)
    #print("imgs shape " + str(imgs.shape))

    for i in range(len(models)):
        probs = models[i].predict(imgs)
        #print("probs shape " + str(probs.shape))
        array_probs[i].append(probs)
    cont = cont+1


array_probs = np.array(array_probs)

print(array_probs.shape)

#save arrays
#output_dir = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/csv_files/inference_arrays/'
#np.save(output_dir+'labels', data_test_labels)

#filename = output_dir+'exp_'+str(i)
#np.save(filename, array_probs)

#METRICS

y_preds = []


def majority(array):
    m = [0,0,0]
    for i in range(len(array)):
        idx = np.argmax(array[i])
        m[idx]=m[idx]+1
    return np.argmax(m)

for i in range(len(models)):
    y_preds.append([])
    for j in range(array_probs.shape[1]):
        pgp = majority(array_probs[i,j,0,:,:])+1
        sgp = majority(array_probs[i,j,1,:,:])+1
        y_preds[i].append([pgp,sgp])
"""
def gleason_score(primary,secondary):
    
    array = []
    for i in range(len(primary)):
        gs = y_true = primary[i]+secondary[i]-2
        
        if (gs==1 and primary[i]==2):
            gs = 2
        elif (gs==1 and primary[i]==1):
            gs = 1
        elif (gs==2):
            gs = 3
        elif (gs>2):
            gs = 4
        array.append(gs)
        
        
    return array
"""

def gleason_score(primary,secondary):
    
    array = []
    for i in range(len(primary)):
        a = primary[i]
        b = secondary[i]
        gs = a+b-2
        
        if (gs==1 and primary[i]==2):
            gs = 2
        elif (gs==1 and primary[i]==1):
            gs = 1
        elif (gs==2):
            gs = 3
        elif (gs>2):
            gs = 4
        array.append(gs)
        
        
    return array   

y_preds = np.array(y_preds)
y_true = data_test_labels

y_preds = y_preds.astype('int64')

def predict_metrics(y_pred,y_true,metric):
    if(metric=='primary'):
        print("y_pred " + str(y_pred.shape))
        #primary gleason pattern
        y_true = y_true[:,0]
        y_pred = y_pred[:,0]
        k_score_primary = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        confusion_matrix_primary = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        cm_normalized_primary = confusion_matrix_primary.astype('float') / confusion_matrix_primary.sum(axis=1)[:, np.newaxis]
        accuracy_balanced_primary = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_score_primary = metrics.f1_score(y_true=y_true, y_pred=y_pred,average='weighted')
        macro_recall_primary = metrics.recall_score(y_true=y_true, y_pred=y_pred,average='macro')
        
        print("k_score_primary " + str(k_score_primary))
        print("confusion_matrix_primary ")
        print(str(confusion_matrix_primary))
        #print('Normalized_confusion_matrix_primary')
        #print(str(cm_normalized_primary))
        print("accuracy_balanced_primary " + str(accuracy_balanced_primary))
        print("f1_score_primary " + str(f1_score_primary))
        print("macro_recall_primary " + str(macro_recall_primary))
        
        return k_score_primary, accuracy_balanced_primary, f1_score_primary, macro_recall_primary
        
    elif(metric=='secondary'):
        #secondary gleason pattern
        y_true = y_true[:,1]
        y_pred = y_pred[:,1]
        k_score_secondary = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        confusion_matrix_secondary = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        cm_normalized_secondary = confusion_matrix_secondary.astype('float') / confusion_matrix_secondary.sum(axis=1)[:, np.newaxis]
        accuracy_balanced_secondary = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_score_secondary = metrics.f1_score(y_true=y_true, y_pred=y_pred,average='weighted')
        macro_recall_secondary = metrics.recall_score(y_true=y_true, y_pred=y_pred,average='macro')
        
        print()
        print("k_score_secondary " + str(k_score_secondary))
        print("confusion_matrix_secondary ")
        print(str(confusion_matrix_secondary))
        #print('Normalized_confusion_matrix_secondary')
        #print(str(cm_normalized_secondary))
        print("accuracy_balanced_secondary " + str(accuracy_balanced_secondary))
        print("f1_score_secondary " + str(f1_score_secondary))
        print("macro_recall_secondary " + str(macro_recall_secondary))
        
        return k_score_secondary, accuracy_balanced_secondary, f1_score_secondary, macro_recall_secondary
        
    else:
        #gleason score
        #y_true = y_true[:,0]+y_true[:,1]
        #y_pred = y_pred[:,0]+y_pred[:,1]

        y_true = gleason_score(y_true[:,0],y_true[:,1])
        y_pred = gleason_score(y_pred[:,0],y_pred[:,1])
        
        
        #print(y_pred)
        k_score_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
        confusion_matrix_score = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        cm_normalized_score = confusion_matrix_score.astype('float') / confusion_matrix_score.sum(axis=1)[:, np.newaxis]
        accuracy_balanced_score = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_score_score = metrics.f1_score(y_true=y_true, y_pred=y_pred,average='weighted')
        macro_recall_score = metrics.recall_score(y_true=y_true, y_pred=y_pred,average='macro')
        
        print()
        print("k_score_score " + str(k_score_score))
        print("confusion_matrix_score ")
        print(str(confusion_matrix_score))
        #print('Normalized_confusion_matrix_score')
        #print(str(cm_normalized_score))
        print("accuracy_balanced_score " + str(accuracy_balanced_score))
        print("f1_score_score " + str(f1_score_score))
        print("macro_recall_score " + str(macro_recall_score))
        
        return k_score_score, accuracy_balanced_score, f1_score_score, macro_recall_score

primaries = []
secondaries = []
scores = []

for i in range(len(y_preds)):
    print("experiment " + str(i))

    kappa_score_primary, _, _, _ = predict_metrics(y_preds[i],y_true,'primary')
    kappa_score_secondary, _, _, _ = predict_metrics(y_preds[i],y_true,'secondary')
    kappa_score_score, _, _, _ = predict_metrics(y_preds[i],y_true,'score')
    primaries.append(kappa_score_primary)
    secondaries.append(kappa_score_secondary)
    scores.append(kappa_score_score)

    for j in range(len(data_test_paths)):
        print_prediction(data_test_paths[j],y_true[j],y_preds[i,j])

primaries = np.array(primaries)
secondaries = np.array(secondaries)
scores = np.array(scores)

avgs_pri = np.mean(primaries)
stds_pri = np.std(primaries)

avgs_sec = np.mean(secondaries)
stds_sec = np.std(secondaries)

avgs_score = np.mean(scores)
stds_score = np.std(scores)

print("avgs ")
print(avgs_pri, avgs_sec, avgs_score)

print("stds ")
print(stds_pri, stds_sec, stds_score)

print("primaries ")
print(primaries)

print("secondaries ")
print(secondaries)

print("scores ")
print(scores)