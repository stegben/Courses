"""

require:
Keras
xgboost

usage: 
python acoustic_frame.py fbank/train.ark label/train.lab fbank/test.ark [submission file name] [nn model name] [supporter model name]
"""



import sys,os
from time import time
import itertools as itr
from random import random
import csv

import numpy as np

import xgboost as xgb

import theano
import theano.tensor as T

from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation, Dropout, MaxoutDense
from keras.layers.advanced_activations import PReLU

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import



train_filename = sys.argv[1]
train_label = sys.argv[2]
test_filename = sys.argv[3]

ftr = open( train_filename , 'r' )
fte = open( test_filename , 'r' )
flab = open( train_label , 'r' )

pred_filename = sys.argv[4]
model_name = sys.argv[5]
modifier_name = sys.argv[6]

pre_frame_num = 2
post_frame_num = 3
holdout = 0.18
epoch = 100

def convolvArray(array,pre,post):
  import numpy as np
  new_array = array
  for i in range(1,pre+1):
    tmp = np.roll( array , i , axis=0)
    new_array = np.concatenate( (tmp , new_array) , axis=1)
  for i in range(1,post+1):
    tmp = np.roll( array , -i , axis=0)
    new_array = np.concatenate( (new_array , tmp) , axis=1)
  return new_array


"""
Process Phones
"""
label_set  = [ 'aa','ae', 'ah','ao', 'aw','ax','ay', 'b',
	      'ch','cl',  'd','dh', 'dx','eh','el','en',
	     'epi','er', 'ey', 'f',  'g','hh','ih','ix',
	      'iy','jh',  'k', 'l',  'm','ng', 'n','ow',
              'oy', 'p',  'r','sh','sil', 's','th','t',
	      'uh','uw','vcl', 'v',  'w', 'y','zh', 'z']

fmap = open('./conf/phones.60-48-39.map' , 'r') # label mapping 48-39
dict_48_39 = {}
dict_60_39 = {}
dict_60_48 = {}
dict_48_ind = {}

for row in fmap:
  l = row.rstrip().split()
  if len(l) < 2: continue
  dict_60_48[ l[0] ] = l[1]
  dict_48_39[ l[1] ] = l[2]
  dict_60_39[ l[0] ] = l[2]

for phone in label_set:
  dict_48_ind[phone] = label_set.index(phone)

flab = open('./label/train.lab'  , 'r')
label_dict = {}
for row in flab:
  lab = row.rstrip().split(",")
  label_dict[lab[0]] = lab[1]

"""
Process X,y for training
"""
# get raw feature dimensions and instance number
name_list = set()
feat_num = set()
row_num = 0
for row in ftr:
  row_num += 1
  feat = row.rstrip().split()
  feat_num.add(len(feat[1:]))
  name_list.add( feat[0].split('_')[0] )

assert len(feat_num) == 1
feat_num = feat_num.pop()
feat_dims = feat_num*( pre_frame_num + post_frame_num+1 )
ftr.seek(0)

cv_name_list = set()
for nl in name_list:
  r = random()
  if r < holdout:
    cv_name_list.add(nl)

# tr_name_list = name_list - cv_name_list

tr_row_num = 0
cv_row_num = 0

for row in ftr:
  if row.rstrip().split()[0].split('_')[0] in cv_name_list:
  	cv_row_num += 1
  else:
  	tr_row_num += 1
ftr.seek(0)


print("creating time-windowed frame...")
t1 = time()

feat_raw = []
Y = np.zeros((tr_row_num,48) , dtype = theano.config.floatX)
Y_cv = np.zeros((cv_row_num,48) , dtype = theano.config.floatX)
X = np.zeros((tr_row_num , feat_num) , dtype = theano.config.floatX)
X_cv = np.zeros((cv_row_num , feat_num) , dtype = theano.config.floatX)

name = []
ind_for_tr = 0
ind_for_cv = 0
for i,row in enumerate(ftr) :
  row = row.rstrip().split(" ")
  feat = [ float(a) for a in row[1:] ]  
  # feat_raw.append( feat )
  # name.append(row[0])
  if row[0].split('_')[0] in cv_name_list:
    X_cv[ ind_for_cv , : ] = feat
    Y_cv[ ind_for_cv , dict_48_ind[label_dict[row[0]]] ] = 1
    ind_for_cv += 1
  else:
    X[ ind_for_tr , : ] = feat
    Y[ ind_for_tr , dict_48_ind[label_dict[row[0]]] ] = 1
    ind_for_tr += 1
  # Y[ i, dict_48_ind[label_dict[row[0]]] ] = 1

X = convolvArray(X , pre_frame_num , post_frame_num)
X_cv = convolvArray(X_cv , pre_frame_num , post_frame_num)

ftr.close()
t2 = time()
print('time-windowed frame creation spend: %0.2f s' % (t2-t1))
del feat_raw

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_cv = scaler.transform(X_cv)


xg_train = xgb.DMatrix( X , label=[y.index(1) for y in Y.tolist()])
del X
del Y
xg_cv = xgb.DMatrix(X_cv , label=[y.index(1) for y in Y_cv.tolist()])

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 1.0
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 6
param['num_class'] = 48
param['subsample'] = 0.7
num_round = 100

eval_list = [(xg_cv , 'eval') , (xg_train , 'train')]

model = xgb.train(param , xg_train , num_round , eval_list , early_stopping_rounds=5)


try:
  for i in range(epoch):
    model.fit(X,Y , batch_size = 128,nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
    model.evaluate(X_cv,Y_cv , show_accuracy=True)
except KeyboardInterrupt:
  print('Stop')


model.save_weights(model_name)

del X
del Y

"""
First prediction, simply by the model just trained
"""
#### get X_test
r = 0
for row in fte:
  r += 1
fte.seek(0)

name = []
X_test = np.zeros( (r , feat_num) , dtype = theano.config.floatX)
for i,row in enumerate(fte) :
  row = row.rstrip().split(" ")
  feat = [ float(a) for a in row[1:] ]  
  X_test[ i , : ] = feat
  name.append(row[0])

X_test = convolvArray(X_test , pre_frame_num , post_frame_num)
X_test = scaler.transform(X_test)
xg_test = xgb.DMatrix(X_test)
ans1 = model.predict(xg_test)


f_pred = open( pred_filename , 'w')


w = csv.writer(f_pred)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans1):
  predictions.append( [n , dict_48_39[ label_set[int(a)] ]] )
w.writerows(predictions)
f_pred.close()


f_pred_48 = open('48_'+pred_filename , 'w')
w = csv.writer(f_pred_48)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans1):
  predictions.append( [n , label_set[a]] )
w.writerows(predictions)
f_pred_48.close()


"""
Second model for supporting correct the answer
by looking at the probability front 2 and back 2
using gradient boosting
"""
prob = model.predict(X_cv)

new_X_cv = convolvArray(prob,5,5) 

sp = cv_row_num * 2 / 3
new_X_cv = np.split(new_X_cv,[sp])
Y_cv = np.split(Y_cv,[sp])



xg_train = xgb.DMatrix( new_X_cv[0], label=[y.index(1) for y in Y_cv[0].tolist()])
xg_cv = xgb.DMatrix(new_X_cv[1] , label=[y.index(1) for y in Y_cv[1].tolist()])

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.15
param['max_depth'] = 5
param['silent'] = 1
param['nthread'] = 6
param['num_class'] = 48
param['subsample'] = 0.7
num_round = 40

eval_list = [(xg_cv , 'eval') , (xg_train , 'train')]

support_model = xgb.train(param , xg_train , num_round , eval_list , early_stopping_rounds=5)
support_model.save_model(modifier_name)


"""
Prediction
"""
prob_test = model.predict(X_test)
new_X_test = convolvArray(prob_test,5,5)
xg_test = xgb.DMatrix(new_X_test)

ans2 = support_model.predict(xg_test)

f_pred = open( 'modified'+pred_filename , 'w')
w = csv.writer(f_pred)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans2):
  predictions.append( [n , dict_48_39[ label_set[int(a)] ]] )
w.writerows(predictions)


f_pred_48 = open('48_modified_'+pred_filename , 'w')
w = csv.writer(f_pred_48)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans2):
  predictions.append( [n , label_set[int(a)]] )
w.writerows(predictions)
