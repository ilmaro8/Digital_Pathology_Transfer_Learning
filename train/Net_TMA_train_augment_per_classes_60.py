import keras
from PIL import Image
#from keras.applications.resnet import ResNet50
#from keras.applications.densenet import DenseNet121, preprocess_input
#from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet,preprocess_input
#from keras.applications.vgg16 import VGG16,preprocess_input
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

np.random.seed(0)

#PATHS
"""
	#if 100%
data_folder = "/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20/train_labels/*"
csv_train = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20/train.csv'
csv_valid = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20/valid.csv'
models_path = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/model_weights/20_patches/20_patches/'
"""

	#if percentage
percentage = '60'
str_perc = str(percentage)
data_folder = "/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20_perc"+str_perc+"/train_labels/*"
csv_train = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20_perc'+str_perc+'/train.csv'
csv_valid = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20/valid.csv'
models_path = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/model_weights/20_patches_'+str_perc+'/'

print("model path " + str(models_path))

target_filename_augment = '/projects/0/examode/examode_experiments/Strong_labels_experiment/TMA_images/all/ZT76_39_A_2_1.jpg'

target = staintools.read_image(target_filename_augment)

print("percentage " + str(str_perc))
 
def interpolate(s_target, s_toaugment, factor):
	s1 = s_target.flatten()
	s2 = s_toaugment.flatten()
	s = np.linspace(s1,s2,factor)
	noise = np.random.normal(0,0.1,6)
	#noise = np.random.rand(6)/10
	idx = np.random.randint(low=0,high=factor)
	s = s[idx] + noise
	s = np.reshape(s, (2, 3))
	return s

class NewStainAugmentor(object):

	def __init__(self, method, sigma1=0.2, sigma2=0.2, augment_background=True, factor=5):
		if method.lower() == 'macenko':
			self.extractor = MacenkoStainExtractor
		elif method.lower() == 'vahadane':
			self.extractor = VahadaneStainExtractor
		else:
			raise Exception('Method not recognized.')
		self.factor = factor
		self.sigma1 = sigma1
		self.sigma2 = sigma2
		self.augment_background = augment_background
		self.image_shape = None
		self.source_concentrations_toaugment = None
		self.stain_matrix_target = None
		self.stain_matrix_toaugment = None
		self.stain_matrix = None
		self.source_concentrations_toaugment = None
		self.n_stains = None
		
	
	def fit_target(self, I):
		"""
		Fit to an image I.
		:param I:
		:return:
		"""
		self.stain_matrix_target = self.extractor.get_stain_matrix(I)

	
	def fit(self, I):
		self.image_shape = I.shape
		self.stain_matrix_toaugment = self.extractor.get_stain_matrix(I)
		self.stain_matrix = interpolate(self.stain_matrix_target,self.stain_matrix_toaugment,self.factor) 
		self.source_concentrations_toaugment = get_concentrations(I, self.stain_matrix)
		#print("concentrations " + str(self.source_concentrations_toaugment))
		self.n_stains = self.source_concentrations_toaugment.shape[1]
		
	
	def pop(self):
		"""
		Get an augmented version of the fitted image.
		:return:
		"""
		augmented_concentrations = copy.deepcopy(self.source_concentrations_toaugment)
		"""
		for i in range(self.n_stains):
			alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
			beta = np.random.uniform(-self.sigma2, self.sigma2)
			if self.augment_background:
				augmented_concentrations[:, i] *= alpha
				augmented_concentrations[:, i] += beta
			else:
				augmented_concentrations[self.tissue_mask, i] *= alpha
				augmented_concentrations[self.tissue_mask, i] += beta
		"""
		
		I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
		I_augmented = I_augmented.reshape(self.image_shape)
		I_augmented = np.clip(I_augmented, 0, 255)

		return I_augmented

colour_augmentor = NewStainAugmentor(method='vahadane', sigma1=0.2, sigma2=0.2)
colour_augmentor.fit_target(target)

# Create your new operation by inheriting from the Operation superclass:
class AugmentColour(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, sigma1, sigma2):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        #self.min_num = min_num
        #self.max_num = max_num
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
    # Your class must implement the perform_operation method:
    def perform_operation(self, image):
        # Start of code to perform custom image operation.
        
        #factor = np.random.randint(low=self.min_num,high=self.max_num)
        
        def do(image):
            img = np.asarray(image[0]).copy()
            #image = cv2.cvtColor(np.array(image))
            augmentor = staintools.StainAugmentor(method='vahadane', sigma1=self.sigma1, sigma2=self.sigma2)
            augmentor.fit(img)
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

folders = []
for f in glob.glob(data_folder):
	if os.path.isdir(f):
		folders.append(os.path.abspath(f))

print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])

tot_elem = 0


pipelines = {}
for folder in folders:
	print("Folder %s:" % (folder))
	pipelines[os.path.split(folder)[1]] = (Augmentor.Pipeline(folder))
	print("\n----------------------------\n")

for p in pipelines.values():
	print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))
	tot_elem = tot_elem+len(p.augmentor_images)

prob = 0.6
for pipeline in pipelines.values():
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
	colour_augment = AugmentColour(probability = prob,sigma1=0.3,sigma2=0.3)
	pipeline.add_operation(colour_augment)


integer_labels = {'0': 0, 
				  '1': 1, 
				  '2': 2, 
				  '3': 3 }


PipelineContainer = collections.namedtuple('PipelineContainer', 
										   'label label_integer label_categorical pipeline generator')

pipeline_containers = []

for label, pipeline in pipelines.items():
	label_categorical = np.zeros(len(pipelines), dtype=np.int)
	label_categorical[integer_labels[label]] = 1
	pipeline_containers.append(PipelineContainer(label, 
												 integer_labels[label], 
												 label_categorical, 
												 pipeline, 
												 pipeline.keras_generator(batch_size=1,scaled=False)))

def check_most_represented(y):
	unique, counts = np.unique(y, return_counts=True,axis=0)
	i = np.argmax(counts)
	max_value = counts[i]
	unique = np.delete(unique, i, 0)
	counts = np.delete(counts, i, 0)
	return unique, counts, max_value

def class_weights_lab(labels):
	unique, counts = np.unique(labels, return_counts=True)
	elems = dict(zip(unique, counts))
	i = np.argmax(counts)
	c_weights = {}

	for j in range(len(unique)):
		c_weights[j] = counts[i]/counts[j]
	print(c_weights)
	return c_weights

data_train = pd.read_csv(csv_train).values
data_valid = pd.read_csv(csv_valid).values

y_train = data_train[:,1]
x_valid = data_valid[:,0]
y_valid = data_valid[:,1]

class_weight = class_weights_lab(y_train)

def load_data(list_f):
	dataset = []
	for f in list_f:
		img = Image.open(f)
		img = np.asarray(img)
		dataset.append(img)
	dataset = np.array(dataset)
	return dataset

print("loading valid data")
X_valid = load_data(x_valid)
print("data loaded")

#PREPROCESS DATA
print("preprocessing valid data")
X_valid = preprocess_input(X_valid)
print("data preprocessed")

from sklearn import preprocessing
y_valid = to_categorical(y_valid,4)


n_samples = 32

def multi_generator(pipeline_containers, batch_size):
	while True:
		X = []
		y = []
		for i in range(batch_size):
			pipeline_container = random.choice(pipeline_containers)
			image, _ = next(pipeline_container.generator)
			image = image.reshape((224,224,3)) # Or (1, 28, 28) for channels_first, see Keras' docs.
			X.append(image)
			y.append(pipeline_container.label_categorical)  # Or label_integer if required by network
		
		#check most representative
		y_, v_, max_value = check_most_represented(y)
		
		#add other classes
		for p in pipeline_containers:
			for i in range(0,len(y_)):
				if np.array_equal(p.label_categorical,y_[i]):
					for j in range(0,max_value-v_[i]):
						image, _ = next(p.generator)
						image = image.reshape((224,224,3)) # Or (1, 28, 28) for channels_first, see Keras' docs.
						X.append(image)
						y.append(p.label_categorical)  # Or label_integer if required by network
		
		X = np.asarray(X)
		X = preprocess_input(X)
		y = np.asarray(y)
		
		yield X, y

batch_size = n_samples

g = multi_generator(pipeline_containers=pipeline_containers, 
					batch_size=batch_size)  # Here the batch size can be set to any value


try:
	pre_trained_model = MobileNet(weights='imagenet')
except:
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	pre_trained_model = MobileNet(weights='imagenet')


"""
#config 6837555: mobilenet, 20 patches per TMA
pre_trained_model.layers.pop()
#fc1 = Dense(512,activity_regularizer=regularizers.l2(0.01))(pre_trained_model.layers[-1].output)
fc1 = Dense(512,activation='relu')(pre_trained_model.layers[-1].output)
d1 = Dropout(0.2)(fc1)
fc2 = Dense(4, activation='softmax')(d1)
model_filename = models_path+'mobilenet_finetune_class.h5'
"""


#CONFIG 6837580, mobilenet, 20 patches per TMA
pre_trained_model.layers.pop()
fc2 = Dense(4, activation='softmax')(pre_trained_model.layers[-1].output)
model_filename = models_path+'mobilenet_finetune.h5'


model = Model(inputs=pre_trained_model.input, outputs=fc2)
model.summary()



try:
	model.load_weights(model_filename)
	print("WEIGHTS LOADED")
except:
	print("weights not found")
	pass

lr = 1e-3

adam = Adam(lr=lr, beta_1=0.9,beta_2=0.999,decay=lr/1000)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history_filename = models_path+'history/history.json'


if not os.path.isdir(models_path):
    try:
        os.mkdir(models_path)
    except OSError:
        print ("Creation of the directory %s failed" % models_path)
    else:
        print ("Successfully created the directory %s " % models_path)


if not os.path.isdir(models_path+'history'):
    try:
        os.mkdir(models_path+'history')
    except OSError:
        print ("Creation of the directory %s failed" % models_path+'history')
    else:
        print ("Successfully created the directory %s " % models_path+'history')

epochs = 20

#MODEL
checkpoint = ModelCheckpoint(filepath=model_filename,
							 monitor='val_categorical_accuracy',
							 verbose=1,
							 save_best_only=True)

start_time = time.time()

#history = model.fit_generator(g ,int(len(y_train) / batch_size),epochs=10, verbose=1, callbacks=[checkpoint], class_weight=class_weight, validation_data=(X_valid,y_valid) ,use_multiprocessing = True, workers = 10, max_queue_size=64)
history = model.fit_generator(g, int(tot_elem / batch_size), epochs=epochs, verbose=1, validation_data=(X_valid,y_valid) ,use_multiprocessing = True, workers = 10, max_queue_size=64, callbacks=[checkpoint], class_weight=class_weight)

elapsed_time = time.time() - start_time
print("done in " + str(elapsed_time))
"""
try:
	model.save(model_filename)
except:
	print("cannot save weights")
"""
try:
	with open(history_filename, 'w') as file:
		json.dump(history.history, file)
except:
	print("cannot save history")

#EVAL

#load data test paper
csv_test = '/projects/0/examode/examode_experiments/Strong_labels_experiment/StrongLabeledNetwork/tma_patches/tma_patches_20/test_paper_patches.csv'

data_test = pd.read_csv(csv_test).values

x_test = data_test[:,0]
y_test = data_test[:,1]


def load_data(list_f):
    dataset = []
    for f in list_f:
        img = Image.open(f)
        img = np.asarray(img)
        dataset.append(img)
    dataset = np.array(dataset)
    return dataset

print("loading data")
X_test = load_data(x_test)
print("data loaded")

#PREPROCESS DATA
print("preprocessing data")
X_test = preprocess_input(X_test)
print("data preprocessed")

from sklearn import preprocessing

y_true = to_categorical(y_test,4)

#TESTING

y_pred = model.predict(x=X_test, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_true, axis=1)

print("y pred " + str(y_pred))
print("y true " + str(y_true) )

#METRICS
#k-score
k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
print("k_score " + str(k_score))
#confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
print("confusion_matrix ")
print(str(confusion_matrix))
#confusion matrix normalized
np.set_printoptions(precision=2)
cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(str(cm_normalized))
#accuracy test
accuracy_balanced = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
print("accuracy_balanced " + str(accuracy_balanced))
#f1_score
f1_score = metrics.f1_score(y_true=y_true, y_pred=y_pred,average='weighted')
print("f1_score " + str(f1_score))

