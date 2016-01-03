import numpy as np
import itertools as itr
import theano
import theano.tensor as T
from time import time
import matplotlib.pyplot as plt
import csv
import os.path
import cPickle as pickle
import random

# function we done by ourselve
from Layer import Layer
from MLP import MLP
import MLPtrainer

######################################################################################################################
#
# parameter setting
#
######################################################################################################################

##### basic parameter
pepoch = 3 # epoch for pretraining
epoch = 400
batch_size = 100
hiddenStruct = [2000,2000,2000,2000]
hiddenActFunc = [2,2,2,2] 
'''
  0: sigmoid
  1: tanh
  2: relu
  3: maxout(not done yet)

  -1: softmax
'''
alpha = 0.001
momentum = 0.95
L1 = 0.0
L2 = 0.0
skip_rate = 0.8
dropout = 0.5
useState = False # use state label(1943) or not(48)
doPT = False # do pre training or not
doDrop = True # do neuron dropout
holdout = 0.02 # cross-validation ratio, 0~1


###### feature parameter
feature = 'fbank' 
pre = 5 # time-window pre frame number
post = 5 # time-window post frame number

###### programming parameter
seePred = False # show training accuracy while in training process. 
               # to get higher speed, set False
showProgress = 40000 # tell user how many instances has been trained


"""
TODOs

Regularization:
1. maxout
2. new normalization

Different Model
1. CNN
2. LSTM

feature preprocessing: 
1. HHT on raw wav, T-F plot
2. fMLLR
3. i-vector

pre-training:
1. DBM
2. in-process pre-training
3. Deep AutoEncoder
 
learning method:
1. NAG
2. AdaDelta descent
3. RMSProp
4. ensemble of the above(use momentum at first, then NAG, then AdaDelta)

"""

##################################################################################################################
#
# Initialization
#
##################################################################################################################

###### stored file name setting
outfilename = 'result_' \
             + feature  \
             + str(pre+post+1) \
             + 'frames' \
             +('_withPT' if doPT else'') \
             + '.csv'
modelName = feature \
          + ('_stateLabel_' if useState else '_') \
          + str(pre+post+1) \
          + 'frames' 
for num in hiddenStruct:
  modelName = modelName + '_' + str(num)
modelName += '.pickle'

##### feature number setting
if feature == 'fbank':
  raw_feat_num = 69
elif feature == 'mfcc':
  raw_feat_num = 39
else :
  raise ValueError('Wrong feature file name')

##### normalization
if os.path.exists(('./std_' + feature + 'Norm.pickle')):
  fnorm = open(('./std_' + feature + 'Norm.pickle') , 'r')
  std , mean = pickle.load(fnorm)
else: 
  ftr = open( ('./' + feature +'/train.ark') , 'r') #training set
  d = [[] for x in range(raw_feat_num)]
  for row in ftr:
    row = row.rstrip().split(" ")
    feat = [ float(a) for a in row[1:] ]
    for h in range(raw_feat_num):
    	d[h].append(feat[h])
  std = [1] * raw_feat_num
  mean = [0] * raw_feat_num
  for ind in range(raw_feat_num):
    mean[ind] = sum(d[ind]) / float(len(d[ind]))
    std[ind] = np.std(np.array(d[ind]))
  ftr.close()
  fnorm = open(('std_' + feature + 'Norm.pickle') , 'w')
  pickle.dump((std,mean) , fnorm)
fnorm.close()


def normalize(x):
  return [(a-m)/b for a,b,m in zip(x,std,mean)]

####### all files used
ftr = open(('./'+ feature +'/train.ark') , 'r') #training set
fte = open(('./'+ feature +'/test.ark')  , 'r') # testing set
if useState:
  flab = open('./state_label/train.lab'  , 'r') # label
else:
  flab = open('./label/train.lab'  , 'r') # label
fmap = open('./phones/state_48_39.map' , 'r') # label mapping 48-39



# label initialization
labelSet  = [ 'aa','ae', 'ah','ao', 'aw','ax','ay', 'b',
        'ch','cl',  'd','dh', 'dx','eh','el','en',
       'epi','er', 'ey', 'f',  'g','hh','ih','ix',
        'iy','jh',  'k', 'l',  'm','ng', 'n','ow',
              'oy', 'p',  'r','sh','sil', 's','th','t',
        'uh','uw','vcl', 'v',  'w', 'y','zh', 'z']
if useState:
  labmap = {}
  for row in fmap:
    l = row.rstrip().split("\t")
    labmap[ l[0] ] = l[2]
else: 
  labmap = {}
  for row in fmap:
    l = row.rstrip().split("\t")
    labmap[ l[1] ] = l[2]
 
###### create label dictionary
label_dict = {}
for row in flab:
  lab = row.rstrip().split(",")
  label_dict[lab[0]] = lab[1]

flab.close()
fmap.close()

#create time-windowed frame
print("creating time-windowed frame...")
t1 = time()
feat_raw = []
name = []
for row in ftr :
  row = row.rstrip().split(" ")
  feat = normalize([ float(a) for a in row[1:] ])
  feat_raw.append( feat )
  name.append(row[0])
feat_new = []
r = len(feat_raw)
for i in range(r):
  new_feat = []
  for k in reversed(range(0,pre)):
    new_feat += feat_raw[i-k-1]
  new_feat += feat_raw[i]
  for h in range(0,post): 
    new_feat += feat_raw[(i+h+1)%r]
  feat_new.append(new_feat)
ftr.close()
t2 = time()
print('time-windowed frame creation spend: %0.2f s' % (t2-t1))

feat_test_raw = []
name = []
for row in fte :
  row = row.rstrip().split(" ")
  feat = normalize([ float(a) for a in row[1:] ])
  feat_test_raw.append( feat )
  name.append(row[0])
feat_test_new = []
r = len(feat_test_raw)
for i in range(r):
  new_feat = []
  for k in reversed(range(0,pre)):
    new_feat += feat_test_raw[i-k-1]
  new_feat += feat_test_raw[i]
  for h in range(0,post): 
    new_feat += feat_test_raw[(i+h+1)%r]
  feat_test_new.append(new_feat)
fte.close()