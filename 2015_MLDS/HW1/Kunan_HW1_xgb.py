"""
2015 Spring MLDS HW1
Apr.9 2015
b00901146 Chu Po-Hsien

use eXtreme Gradient Bossting
"""

import numpy as np
import itertools as itr
from time import time
import matplotlib.pyplot as plt
import csv
import os.path
import cPickle as pickle
import random

import xgboost as xgb


######################################################################################################################
#
# parameter setting
#
######################################################################################################################

##### basic parameter
pepoch = 3 # epoch for pretraining
epoch = 400
batch_size = 100
alpha = 5
L1 = 0.0
L2 = 0.0
# skip_rate = 0.7
useState = False # use state label(1943) or not(48)
holdout = 0.1 # cross-validation ratio, 0~1

###### feature parameter
feature = 'fbank' 
pre = 3 # time-window pre frame number
post = 3 # time-window post frame number

###### programming parameter
seePred = False # show training accuracy while in training process. 
               # to get higher speed, set False
showProgress = 40000 # tell user how many instances has been trained

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
             + '.csv'
"""
modelName = feature \
          + ('_stateLabel_' if useState else '_') \
          + str(pre+post+1) \
          + 'frames' 
for num in hiddenStruct:
  modelName = modelName + '_' + str(num)
modelName += '.pickle'
"""
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

######## split training and CV
name_list = []
for n in name:
  tmp = n.split("_")
  if tmp[0] not in name_list:
    name_list.append(tmp[0])

cv_name_list = []
for nl in name_list:
  r = random.random()
  if r < holdout:
    cv_name_list.append(nl)

tr_name_list = list(set(cv_name_list)^set(name_list))

cv_name , cv_feat_new, tr_name , tr_feat_new = [] , [] , [] ,[]

for n,f in itr.izip(name,feat_new):
  if n.split("_")[0] in cv_name_list:
    cv_name.append(n)
    cv_feat_new.append(f)

for n,f in itr.izip(name,feat_new):
  if n.split("_")[0] in tr_name_list:
    tr_name.append(n)
    tr_feat_new.append(f)


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

"""
train_X = np.array(tr_feat_new)
train_Y = np.array([labelSet.index(label_dict[n]) for n in tr_name ])
cv_X    = np.array(cv_feat_new)
cv_Y    = np.array([labelSet.index(label_dict[n]) for n in cv_name ])
"""
xg_train = xgb.DMatrix(np.array(tr_feat_new)
                 , label = np.array([labelSet.index(label_dict[n]) for n in tr_name ]))
xg_cv    = xgb.DMatrix(np.array(cv_feat_new)
                 , label = np.array([labelSet.index(label_dict[n]) for n in cv_name ]))

param['max_depth']=2
param['eta']=1
param['silent']=0
param['nthread'] = 2
param['num_class'] = 6
param['subsample'] = 0.5
param['gamma'] = 0
# param['eval_metric'] = 'mlogloss'
param['seed'] = 123456

#param['objective']='binary:logistic'
#param['objective']='multi:softmax'
param['objective'] = 'multi:softprob'

num_round = epoch

watch_list = [ (xg_train , 'train') , (xg_cv , 'eval') ]

bst = xgb.train(param , xg_train , num_round , watch_list , early_stopping_rounds=10)

bst.save_model('test.model')


###### write in file preparation
fresult = open(outfilename , 'w')
w = csv.writer(fresult)
w.writerow(['Id' , 'Prediction'])

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

test_X = np.array(feat_test_new)
xg_test = xgb.DMatrix(test_X)
ypred = bst.predict(xg_test)


"""
###### start predicting 
i = 0
print('start predicting...')
for n,row in itr.izip(name,feat_test_new):
  
  # feature vector
  input = np.array([row] , dtype = theano.config.floatX)
  
  # get prediction distribution
  out = pred(input)
  largestInd = max( (v, i) for i, v in enumerate(out[0]) )[1]
  if useState:
    label = labmap[ str(largestInd) ]
  else:
    label = labmap[ labelSet[largestInd] ]
  
  result = []
  result.append(n) # id
  result.append(label) # label
  
  w.writerow(result)

  if i % showProgress == 0:
    print('test instances number: %d' % i)
  i += 1

print('predicting done.')
netFile = open(modelName , 'w')
pickle.dump(model,netFile)
fresult.close()
plt.plot(error)
plt.show()
"""