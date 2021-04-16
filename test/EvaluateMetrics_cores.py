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
import os
from PIL import Image
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ProgbarLogger
from sklearn import metrics
from keras import regularizers


# In[2]:


#PATHS
models_path = '/home/niccolo/ExamodePipeline/JHBI_review/model_weights/inference/'
data_folder = '/home/niccolo/ExamodePipeline/TMA_patches/'

csv_test = data_folder+'test_paper_patches.csv'

csv_ground_truth = data_folder+'ZT80_gleason_scores.csv'

tma_test_patches_dir = '/mnt/nas4/datasets/ToReadme/prostate_TMAs/dataverse_files/TMA_images/test_patches_224/patho_1/'


# In[3]:


def load_model(filename):
    pre_trained_model = MobileNet(weights='imagenet')
    pre_trained_model.layers.pop()
    fc1 = Dense(512,activity_regularizer=regularizers.l2(0.01))(pre_trained_model.layers[-1].output)
    d1 = Dropout(0.2)(fc1)
    fc2 = Dense(4, activation='softmax',activity_regularizer=regularizers.l2(0.01))(d1)
    model = Model(inputs=pre_trained_model.input, outputs=fc2)
    
    try:
        model.load_weights(filename)
        print("WEIGHTS LOADED")
    except:
        print("weights not found")
        pass
    
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    
    return model


# In[4]:


def predict_metrics(model,X_test,y_true):
    y_pred = model.predict(x=X_test, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    
    #kappa score
    k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
    print("k_score " + str(k_score))
    accuracy_balanced = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    print("accuracy_balanced " + str(accuracy_balanced))
    #f1_score
    f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred,average='weighted')
    print("f1_score " + str(f1_score))
    #macro_recall
    macro_recall = metrics.recall_score(y_true=y_true, y_pred=y_pred,average='macro')
    print("macro_recall " + str(macro_recall))
    return k_score, accuracy_balanced, f1_score, macro_recall


# In[5]:


#LOAD DATA
ground_truth = pd.read_csv(csv_ground_truth).values
list_test_data = os.listdir(tma_test_patches_dir)[:-1]


# In[6]:


def find_first_two(array):
    x = np.copy(array)
    max_1 = np.argmax(x)
    max_val1 = x[max_1]
    x[max_1]=-1
    max_2 = np.argmax(x)
    max_val2 = x[max_2]
    if (max_val1>(2*max_val2)):
        max_2 = max_1
    return max_1,max_2


# In[7]:


def majority_voting(array):
    majority = []
    for i in range(len(array)):
        i1,i2 = find_first_two(array[i])
        majority.append([i1,i2])
    majority = np.array(majority)
    return majority


# In[8]:

"""
def assign_group(a, b, survival_groups=False):
    # if both cancer and benign tissue are predicted
    # ignore benign tissue for reporting, as pathologists do
    if (a > 0) and (b == 0):
        b = a
    if (b > 0) and (a == 0):
        a = b

    if not survival_groups:
        return a + b
    else:
        # get the actual Gleason pattern (range 3-5)
        a += 2
        b += 2
        if a+b <= 6:
            return 1
        elif a+b == 7:
            return 2
        else:
            return 3
"""

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

# In[9]:


def pathologist_evaluation(array):
    new_evaluation = []
    for i in range(len(array)):
        gg = assign_group(array[i,0],array[i,1])
        new_evaluation.append(gg)
    new_evaluation = np.array(new_evaluation)
    return new_evaluation


# In[10]:


def predict_metrics(model,y_pred,y_true,metric):
    if(metric=='PGP'):
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
        
    elif(metric=='SGP'):
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
        
        return k_score_secondary, accuracy_balanced_secondaryy, f1_score_secondary, macro_recall_secondary
        
    else:
        #gleason score
        #y_true = y_true[:,0]+y_true[:,1]
        #y_pred = y_pred[:,0]+y_pred[:,1]
        
        y_true = pathologist_evaluation(y_true)
        y_pred = pathologist_evaluation(y_pred)
        
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
        
        
        


# In[11]:


def gen_data(model):
    test_data = []
    labels = []
    for i in range(len(list_test_data)):
        core = []
        cur_dir = tma_test_patches_dir+list_test_data[i]+'/'
        list_files = os.listdir(cur_dir)
        for f in list_files:
            filename = cur_dir+f
            img = Image.open(filename)
            img = np.asarray(img)
            patch = preprocess_input(img) 
            patch = np.expand_dims(patch,axis=0)
            probs = model.predict(x=patch, verbose=0)
            probs = probs.flatten()
            core.append(probs)
        core = np.sum(core,axis=0)
        if(isinstance(core, np.ndarray)):
            for j in range(len(ground_truth)):
                if (list_test_data[i]==ground_truth[j,0]):
                    labels.append([ground_truth[j,1],ground_truth[j,2]])
                    test_data.append(core)

    test_data = np.array(test_data)
    labels = np.array(labels)

    test_data = majority_voting(test_data)
    return test_data,labels


# In[12]:


#load models
percentages = ['20','40','60','80','100']
#percentages = ['100']
experiments = ['0','1','2','3','4','5','6','7','8','9']
#experiments = ['8']

#metric = 'PGP'
#metric = 'SGP'
metric = 'GS'

kappas = []
cumulative_kappas = []
f1_scores = []
cumulative_f1_scores = []
b_accuracies = []
cumulative_b_accuracies = []
macro_recalls = []
cumulative_macro_recalls = []


for i in range(len(percentages)):
    print('percentage ' + str(percentages[i]))
    kappas = []
    f1_scores = []
    b_accuracies = []
    macro_recalls = []
    for j in range(len(experiments)):
        print('experiments ' + str(experiments[j]))
        exp_folder = models_path+'exp_'+experiments[j]+'/'+percentages[i]+'_perc/mobilenet_finetune_class.h5'
        #print(exp_folder)
        model = load_model(exp_folder)
        
        test_data,labels = gen_data(model)
        
        k_score, accuracy_balanced, f1_score, macro_recall = predict_metrics(model,test_data,labels,metric)
        kappas.append(k_score)
        f1_scores.append(f1_score)
        b_accuracies.append(accuracy_balanced)
        macro_recalls.append(macro_recall)

        filename_savefile = models_path+'exp_'+experiments[j]+'/'+percentages[i]+'_perc/checkpoint_dir/kappa_score_'+metric+'_cores_TMAZ.csv'

        File = {'val':[k_score]}
        df = pd.DataFrame(File,columns=['val'])
        np.savetxt(filename_savefile, df.values, fmt='%s',delimiter=',')
        
    cumulative_kappas.append(kappas)
    cumulative_f1_scores.append(f1_scores)
    cumulative_b_accuracies.append(b_accuracies)
    cumulative_macro_recalls.append(macro_recalls)


# In[13]:


print(cumulative_kappas)


# In[14]:


avgs = []
stds = []
for i in range(len(cumulative_kappas)):
    avgs.append(np.mean(cumulative_kappas[i]))
    stds.append(np.std(cumulative_kappas[i]))



print(avgs, stds)






