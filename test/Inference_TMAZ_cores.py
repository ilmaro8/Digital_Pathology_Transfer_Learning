from PIL import Image
from torchvision import transforms
import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import copy
from sklearn import metrics
import sys, getopt
import os
import glob
import random
import collections
import time
from tqdm import tqdm

argv = sys.argv[1:]

np.random.seed(0)

models_path = '/home/niccolo/ultrafast/ExamodePipeline/Semi_Supervised_Learning/models_weights/'
data_folder = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/TMA_core_inference/test_paper_folder_pre/'

csv_test = data_folder+'test_paper_patches.csv'

csv_ground_truth = data_folder+'ZT80_gleason_scores.csv'

tma_test_patches_dir = data_folder+'patho_1/'

try:
    opts, args = getopt.getopt(argv,"hn:p:s:",["n_exp=","pre_trained=","subset="])
except getopt.GetoptError:
    print('Student_Model_finetuning_var_pretrained.py -n <n_exp> -p <pre_trained> -s <subset>' )
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Student_Model_finetuning_var_pretrained.py -n <n_exp> -p <pre_trained> -s <subset>')
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class StudentModel(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(StudentModel, self).__init__()
        self.pretrained = torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=imageNet_weights)

        self.pretrained.classifier = torch.nn.Sequential(
                         torch.nn.Linear(in_features=1024,out_features=4))
                         #torch.nn.Softmax())
        self.pretrained = torch.nn.DataParallel(self.pretrained)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        output=self.pretrained(x)
        #output = self.head(x)
        return output


#DATA NORMALIZATION
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#load_models
if (EXPERIMENT=='fully'):
    model_path = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/student_model_baseline/N_EXP_'+N_EXP_str+'/student_model_finetuning.pt'
    model = torch.load(model_path)
    checkpoint_dir = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/student_model_baseline/N_EXP_'+N_EXP_str+'/checkpoints_pre_training/'
elif (EXPERIMENT=='semi_selected_pre'):
    model_path = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/selected/student_model_pre_training/perc_'+str(SUBSET)+'/N_EXP_'+N_EXP_str+'/student_model_pre_training.pt'
    model = torch.load(model_path)
    checkpoint_dir = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/selected/student_model_pre_training/perc_'+SUBSET+'/N_EXP_'+N_EXP_str+'/checkpoints_pre_training/'
elif (EXPERIMENT=='semi_selected_fine'):
    model_path = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/selected/student_model_finetuning/perc_'+str(SUBSET)+'/N_EXP_'+N_EXP_str+'/student_model_finetuning.pt'
    model = torch.load(model_path)
    checkpoint_dir = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/selected/student_model_finetuning/perc_'+SUBSET+'/N_EXP_'+N_EXP_str+'/checkpoints_pre_training/'
elif (EXPERIMENT=='semi_selected_combined'):
    model_path = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/combining_datasets/selected/student_model_combining/perc_'+str(SUBSET)+'/N_EXP_'+N_EXP_str+'/student_model_finetuning.pt'
    model = torch.load(model_path)
    checkpoint_dir = '/home/niccolo/ExamodePipeline/Semi_Supervised_Learning/models_weights/student_models/combining_datasets/selected/student_model_combining/perc_'+SUBSET+'/N_EXP_'+N_EXP_str+'/checkpoints_pre_training/'


model.eval()
model.to(device)

def create_dir(directory):
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except OSError:
            print ("Creation of the directory %s failed" % directory)

class Dataset_TCGA_test(data.Dataset):

    def __init__(self, list_IDs, cur_dir):

        self.list_IDs = list_IDs
        self.cur_dir = cur_dir
    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        # Select sample
        ID = self.list_IDs[index]
        #print(ID)
        # Load data and get label
        ID = self.cur_dir+ID
        X = Image.open(ID)
        X = np.asarray(X)

        #data transformation
        input_tensor = preprocess(X)
                
        return input_tensor

batch_size = 100
num_workers = 10
params_test = {'batch_size': batch_size,
          #'shuffle': True,
          #'sampler': ImbalancedDatasetSampler(test_dataset),
          'num_workers': num_workers}

#LOAD DATA
ground_truth = pd.read_csv(csv_ground_truth).values
list_test_data = os.listdir(tma_test_patches_dir)#[:-1]

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

def majority_voting(array):
    majority = []
    for i in range(len(array)):
        i1,i2 = find_first_two(array[i])
        majority.append([i1,i2])
    majority = np.array(majority)
    return majority

def assign_group(a, b, survival_groups=False):
    # if both cancer and benign tissue are predicted
    # ignore benign tissue for reporting, as pathologists do
    if (a > 0) and (b == 0):
        b = a
    if (b > 0) and (a == 0):
        a = b
    #print(a,b)
    if not survival_groups:
        return a + b
    else:
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

        """
        a += 2
        b += 2
        if a+b <= 6:
            return 1
        elif a+b == 7:
            return 2
        else:
            return 3
        """
def pathologist_evaluation(array):
    new_evaluation = []
    for i in range(len(array)):
        gg = assign_group(array[i,0],array[i,1],True)
        new_evaluation.append(gg)
    new_evaluation = np.array(new_evaluation)
    return new_evaluation

def predict_metrics(model,y_pred,y_true,metric):
    if(metric=='primary'):
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
        
        print("k_score_score " + str(k_score_score))
        print("confusion_matrix_score ")
        print(str(confusion_matrix_score))
        #print('Normalized_confusion_matrix_score')
        #print(str(cm_normalized_score))
        print("accuracy_balanced_score " + str(accuracy_balanced_score))
        print("f1_score_score " + str(f1_score_score))
        print("macro_recall_score " + str(macro_recall_score))
        
        return k_score_score, accuracy_balanced_score, f1_score_score, macro_recall_score

def get_histogram_ground_truth(listfiles):
    histogram = [0,0,0,0]
    for f in listfiles:
        idx = int(f[-5])
        histogram[idx] = histogram[idx]+1 
    return histogram

def get_histogram_preds(probs):
    histogram = [0,0,0,0]
    for p in probs:
        idx = np.argmax(p)
        histogram[idx] = histogram[idx]+1 
    return histogram

def gen_data(model):
    test_data = []
    labels = []

    print("len list " + str(len(list_test_data)))
    for i in range(len(list_test_data)):
    #for i in range (5):
        core = []
        cur_dir = tma_test_patches_dir+list_test_data[i]+'/'
        list_files = os.listdir(cur_dir)
        
        testing_set = Dataset_TCGA_test(list_files,cur_dir)
        testing_generator = data.DataLoader(testing_set, **params_test)
        array_probs = []
        
        with torch.no_grad():
            j = 0
            for inputs in testing_generator:
                inputs = inputs.to(device)
                # zero the parameter gradients
                #optimizer.zero_grad()

                # forward + backward + optimize
                try:
                    outputs = model(inputs)
                except:
                    outputs = torch.nn.DataParallel(model.pretrained.module)(inputs)
                probs = F.softmax(outputs)
                #print(probs)

                #accumulate values
                probs = probs.cpu().data.numpy()
                array_probs = np.append(array_probs,probs)
                core.append(probs)   
                
        histogram_ground_truth = get_histogram_ground_truth(list_files)        
        histogram_preds = get_histogram_preds(probs)
        #print("filename " + list_test_data[i] + " histogram pred " + str(histogram_preds) + " histogram ground_truth " + str(histogram_ground_truth))

        filenames.append(list_test_data[i])
        histo_true.append(histogram_ground_truth)
        histo_preds.append(histogram_preds)

        #print(histogram_ground_truth)
        #print(histogram_preds)
        
        core = np.sum(core,axis=0)

        if(isinstance(core, np.ndarray)):
            for j in range(len(ground_truth)):
                if (list_test_data[i]==ground_truth[j,0]):
                    labels.append([ground_truth[j,1],ground_truth[j,2]])
                    test_data.append(find_first_two(histogram_preds))

    test_data = np.array(test_data)
    labels = np.array(labels)

    return test_data,labels

#metric = 'primary'
#metric = 'secondary'
metric = 'score'

kappas = []
f1_scores = []
b_accuracies = []
macro_recalls = []

filenames = []
histo_preds = []
histo_true = []
     
model.eval()
model.to(device)
test_data,labels = gen_data(model)
        
k_score, accuracy_balanced, f1_score, macro_recall = predict_metrics(model,test_data,labels,metric)
#print("kappa_score " + str(k_score))

kappa_score_best_GS_filename = checkpoint_dir+'kappa_score_GS_cores_TMAZ.csv'

kappas = [k_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_best_GS_filename, df.values, fmt='%s',delimiter=',')