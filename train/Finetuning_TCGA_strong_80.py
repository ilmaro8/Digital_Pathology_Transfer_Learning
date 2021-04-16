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
import Augmentor
#colour augmentation
from Augmentor.Operations import Operation
import staintools
import json
import copy
from sklearn import metrics
import os
import glob
import random
import collections

import time

from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
from staintools.miscellaneous.get_concentrations import get_concentrations

from keras import regularizers
from tqdm import tqdm

np.random.seed(0)

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

N_EXP = '9'

#PATHS


	#if 100%
csv_train = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/csv_files/finetuning_csv/train_finetuning.csv'
csv_valid = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/csv_files/finetuning_csv/valid_finetuning.csv'
csv_test = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/csv_files/finetuning_csv/test_finetuning.csv'
models_path = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/model_weights/finetuning/strong_finetuning_80/exp_'+N_EXP+'/'

if not os.path.isdir(models_path):
    try:
        os.mkdir(models_path)
    except OSError:
        print ("Creation of the directory %s failed" % models_path)
    else:
        print ("Successfully created the directory %s " % models_path)

print("model path " + str(models_path))

# Create your new operation by inheriting from the Operation superclass:
class AugmentColour(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, sigma1, sigma2,augmentor):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        #self.min_num = min_num
        #self.max_num = max_num
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augmentor = augmentor
        
    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        # Start of code to perform custom image operation.
        
        #factor = np.random.randint(low=self.min_num,high=self.max_num)
        
        def do(image):
            img = np.asarray(image[0]).copy()
            #image = cv2.cvtColor(np.array(image))
            #augmentor = staintools.StainAugmentor(method='vahadane', sigma1=self.sigma1, sigma2=self.sigma2)
            self.augmentor.fit(img)
            return augmentor.pop()

        augmented_images = []
        augmented_images.append(do(image))
        return augmented_images

# Create your new operation by inheriting from the Operation superclass:
class NewAugmentColour(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, augmentor):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        #self.min_num = min_num
        #self.max_num = max_num
        self.augmentor = augmentor

    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        # Start of code to perform custom image operation.
        
        #factor = np.random.randint(low=self.min_num,high=self.max_num)
        
        def do(image):
            try:
                img = np.asarray(image[0]).copy()
                #image = cv2.cvtColor(np.array(image))
                augmentor.fit(img)
                new_img = augmentor.pop()
            except:
                #print("error tissue")
                new_img = np.asarray(image[0]).copy()
            return new_img
        
        augmented_images = []
        augmented_images.append(do(image))
        
        #for _ in range(factor):
        #    augmented_images.append(do(image))
        # End of code to perform custom image operation.

        # Return the image so that it can further processed in the pipeline:
        return augmented_images

augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.3, sigma2=0.3)

def get_pipeline(data,labels):
    global augmentor
    prob = 0.5
    
    pipeline = Augmentor.DataPipeline(data,labels)

    pipeline.rotate_random_90(prob)

    # First, we add a horizontal flip operation to the pipeline:
    pipeline.flip_left_right(probability=prob)
    # Now we add a vertical flip operation to the pipeline:
    pipeline.flip_top_bottom(probability=prob)

    #shear

    #morphological
    pipeline.random_distortion(probability=prob, grid_width=5, grid_height=5, magnitude=8)
    #pipeline.gaussian_distortion(probability=prob, grid_width=4, grid_height=4, magnitude=8, corner='bell', method='in', mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)

    #colour
    #colour_augment = NewAugmentColour(probability = prob, augmentor = colour_augmentor)
    colour_augment = AugmentColour(probability = prob,sigma1=0.5,sigma2=0.5, augmentor=augmentor)
    pipeline.add_operation(colour_augment)

    return pipeline

#LOAD DATA and LABELS
data_train = pd.read_csv(csv_train,header=None).values
data_valid = pd.read_csv(csv_valid,header=None).values
data_test = pd.read_csv(csv_test,header=None).values

def load_data(list_f):
    dataset = []
    for i in tqdm(range(len(list_f))):
        img = Image.open(list_f[i])
        img = np.asarray(img)
        dataset.append(img)
    dataset = np.array(dataset)
    return dataset

#def generators
x_copy_train = np.copy(data_train)
x_copy_valid = np.copy(data_valid)

def multi_generator(batch_size,reset_size,array):
    x = np.copy(array)
    cont = 0
    while True:
        
        if (cont>reset_size):
            x = np.copy(array)
            cont = 0
        
        X = []
        y1 = []
        y2 = []

        if(len(x)>batch_size):
            idxs = np.random.choice(len(x), batch_size,replace=False)
        else:
            idxs = np.arange(len(x))
        
        #DATA AUGMENT

        for i in idxs:

            lab1 = [0,0,0]
            lab2 = [0,0,0]
            path = x[i,0]
            #open img
            img = Image.open(path)
            img = np.asarray(img)

            label1 = x[i,1]-1
            label2 = x[i,2]-1

            lab1[label1]=1
            lab2[label2]=1


            X.append(img)
            y1.append(lab1)
            y2.append(lab2)
        
        
        x = np.delete(x, idxs, 0)
        
        X = np.asarray(X)
        X = preprocess_input(X)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)

        cont = cont+1
                    
        
        yield X, [y1, y2]

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

def check_most_represented(y):
    unique, counts = np.unique(y, return_counts=True,axis=0)
    #elems = dict(zip(unique, counts))
    i = np.argmax(counts)
    gs = unique[i]
    max_value = counts[i]
    #print(unique,counts)
    unique = np.delete(unique, i, 0)
    counts = np.delete(counts, i, 0)
    return unique, counts, max_value, gs

def generate_gleason_dict():
    gleason_dicts = []
    for i in range(5):
        d = {"gs": i, "paths": [], "labs": []}
        gleason_dicts.append(d)
    return gleason_dicts

def add_dict(n,dict_gs,path,labels):
    b = False
    i = 0
    while(b==False and i<len(dict_gs)):
        if(dict_gs[i]["gs"]==n):
            b = True
            dict_gs[i]["paths"].append(path)
            dict_gs[i]["labs"].append(labels)
        i = i+1

    return dict_gs

def multi_generator_augmentor(batch_size,reset_size,array):
    x = np.copy(array)
    cont = 0
    while True:
        
        if (cont>reset_size):
            x = np.copy(array)
            cont = 0
            
        gleason_dict = generate_gleason_dict()
        
        X = []
        y1 = []
        y2 = []

        if(len(x)>batch_size):
            idxs = np.random.choice(len(x), batch_size,replace=False)
        else:
            idxs = np.arange(len(x))
        paths = []
        labels_aug = []
        gleasons = []
        #DATA AUGMENT
        
        for i in idxs:
            lab1 = [0,0,0]
            lab2 = [0,0,0]
            path = x[i,0]
            #open img            
            
            img = Image.open(path)
            img = np.asarray(img)
            img = img.reshape((224,224,3))
            label1 = x[i,1]-1
            label2 = x[i,2]-1
            
            lab1[label1]=1
            lab2[label2]=1
            
            X.append(img)
            y1.append(lab1)
            y2.append(lab2)
            
            
            paths.append((path,path))
            labels_aug.append((label1,label2))
            gs = gleason_score(x[i,1],x[i,2])
            gleasons.append(gs)
            gleason_dict = add_dict(gs,gleason_dict,(path,path),(label1,label2))
            
        
        y_, v_, max_value, gs_max = check_most_represented(gleasons)
        #print(y_, v_, max_value, gs_max)
        
        j = 0
        for i in range(len(gleason_dict)):
            if (gleason_dict[i]["gs"]!=gs_max and i in y_):
                images = [[np.asarray(Image.open(y)) for y in x] for x in gleason_dict[i]["paths"]]
                aug_batch_size = max_value-v_[j]
                
                pipeline = get_pipeline(images,gleason_dict[i]["labs"])
                g = pipeline.generator(batch_size=aug_batch_size)
                
                list_img, labs = next(g)
                #print(labs)
                for k in range(aug_batch_size):
                    lab1 = [0,0,0]
                    lab2 = [0,0,0]
                    
                    idx_1 = labs[k][0]
                    idx_2 = labs[k][1]

                    lab1[idx_1]=1
                    lab2[idx_2]=1

                    X.append(list_img[k][0].reshape((224,224,3)))
                    y1.append(lab1)
                    y2.append(lab2)
                j = j+1
        
        x = np.delete(x, idxs, 0)
        
        X = np.asarray(X)
        X = preprocess_input(X)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        
        #check most representative
             
        
        cont = cont+1
                    
        
        yield X, [y1, y2]


epochs = 5
batch_size = 32

train_steps = len(data_train)//batch_size
#train_steps = 5
val_steps=len(data_valid)//batch_size
#val_steps = 5

train_gen = multi_generator_augmentor(batch_size,train_steps,data_train)
valid_gen = multi_generator(batch_size,val_steps,data_valid)

#train_gen = multi_generator(batch_size,train_steps,data_train)
#valid_gen = multi_generator(batch_size,val_steps,data_valid)

#MODEL
try:
	pre_trained_model = MobileNet(weights='imagenet')
except:
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	pre_trained_model = MobileNet(weights='imagenet')

#config 6837555: mobilenet, 20 patches per TMA
pre_trained_model.layers.pop()
#fc1 = Dense(512,activity_regularizer=regularizers.l2(0.01))(pre_trained_model.layers[-1].output)
fc1 = Dense(512)(pre_trained_model.layers[-1].output)
d1 = Dropout(0.2)(fc1)
fc2 = Dense(4, activation='softmax',activity_regularizer=regularizers.l2(0.01))(d1)
model_filename_base = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/model_weights/exp_1/80_perc/mobilenet_finetune_class.h5'

p_model = Model(inputs=pre_trained_model.input, outputs=fc2)

try:
    p_model.load_weights(model_filename_base)
    print("WEIGHTS LOADED")
except:
    print("weights not found")
    pass

#primary classifier
x1 = Dense(3, activation="softmax", name='output_1')(d1)

#secondary classifier
x2 = Dense(3, activation="softmax", name='output_2')(d1)

model = Model(inputs=p_model.input, outputs=[x1, x2])

model_filename = models_path+'mobilenet_finetune_WSI_class_strong.h5'

#MODEL
checkpoint1 = ModelCheckpoint(filepath=model_filename,
                             monitor='val_output_1_acc',
                             verbose=1,
                             save_best_only=True)

checkpoint2 = ModelCheckpoint(filepath=model_filename,
                             monitor='val_output_2_acc',
                             verbose=1,
                             save_best_only=True)



opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-3 / 100)
model.compile(loss={'output_1': 'categorical_crossentropy', 'output_2': 'categorical_crossentropy'},optimizer=opt, metrics=['acc'])
model.summary()
try:
    model.load_weights(model_filename)
    print("WEIGHTS LOADED")
except:
    print("weights not found")
    pass

history_filename = models_path+'history/history_finetune_strong.json'
if not os.path.isdir(models_path+'history'):
    try:
        os.mkdir(models_path+'history')
    except OSError:
        print ("Creation of the directory %s failed" % models_path+'history')
    else:
        print ("Successfully created the directory %s " % models_path+'history')


#TRAINING

history = model.fit_generator(train_gen, train_steps, shuffle=True, epochs=epochs, verbose=1, validation_data=valid_gen, validation_steps=val_steps,use_multiprocessing = True, workers = 5, max_queue_size=64, callbacks=[checkpoint1,checkpoint2])#, class_weight=class_weight)
#history = model.fit_generator(train_gen, train_steps, shuffle=True, epochs=epochs, verbose=1, use_multiprocessing = True, workers = 10, max_queue_size=64, callbacks=[checkpoint1,checkpoint2])#, class_weight=class_weight)

try:
    with open(history_filename, 'w') as file:
        json.dump(history.history, file)
except:
    print("cannot save history")
#TESTING
