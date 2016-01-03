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

# import xgboost as xgb

import theano
import theano.tensor as T

from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, MaxoutDense,Reshape, TimeDistributedDense
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import GRU, LSTM

from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import



train_filename = sys.argv[1]
train_label = sys.argv[2]
# test_filename = sys.argv[3]

ftr = open( train_filename , 'r' )
# fte = open( test_filename , 'r' )
flab = open( train_label , 'r' )

# pred_filename = sys.argv[4]
model_name = sys.argv[3]


pre_frame_num = 2
post_frame_num = 2
holdout = 0.1
epoch = 15

## ( sequence number , batch size , epoch )
repeat_num = [(5,64,2),(8,40,2),(14,30,3),(30,15,5),(50,10,3)]

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


def fold(X , repeat_num , feat_dims):
  pad = repeat_num - len(X) % repeat_num
  pad_f = pad // 2
  pad_p = pad - pad_f

  X = np.lib.pad(X    , ((pad_f , pad_p) ,(0,0) ) , 'edge' )
  X = X.reshape(len(X)/repeat_num , repeat_num , feat_dims)
  
  return (X , pad_f , pad_p)

def fold_back( X , r_n , feat_dims , pad_f , pad_p ):
  X = X.reshape(X.shape[0]*r_n , feat_dims)
  X = X[ pad_f:(X.shape[0]-pad_p) , :]
  return X

def getResultByDiffSeq(model , X , repeat_num , feat_dims , reverse=False):
  result = []
  for r_n in repeat_num:
    tmpX , f , p = fold(X , r_n[0] , feat_dims)
    if reverse:
      tmpX = np.fliplr(tmpX)
    pred = model.predict(tmpX,batch_size=r_n[1])
    pred = fold_back( pred , r_n[0] , 48 , f , p )
    result.append(pred)
  Y = np.mean( np.array(result) , axis=0 )
  return Y


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

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_cv = scaler.transform(X_cv)

t2 = time()
print('time-windowed frame creation spend: %0.2f s' % (t2-t1))
del feat_raw


# print(X)
# print(revX)

"""
Model 1
RNN Keras Model for direct transform
"""
model = Sequential()

# model.add(TimeDistributedDense(feat_dims,2048,activation='relu'))
#model.add(PReLU(2048))
# model.add(Dropout(0.4))

# model.add(MaxoutDense(2048,256,nb_feature=8))
# model.add(Dropout(0.4))

model.add( GRU(feat_dims, 512 , return_sequences=True, activation='tanh') )
model.add(Dropout(0.2))

model.add( GRU(512, 512 , return_sequences=True, activation='tanh') )
model.add(Dropout(0.2))

model.add(TimeDistributedDense(512,2048,activation='relu'))
#model.add(PReLU(2048))
model.add(Dropout(0.4))

model.add(TimeDistributedDense(2048,1024,activation='relu'))
#model.add(PReLU(1024))
model.add(Dropout(0.4))

model.add(TimeDistributedDense(1024,48))
model.add(Activation('time_distributed_softmax'))

trainer = Adadelta(lr = 1.0 , rho = 0.97 , epsilon = 1e-8 )
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)


revmodel = Sequential()

# revmodel.add(TimeDistributedDense(feat_dims,2048,activation='relu'))
#revmodel.add(PReLU(2048))
# revmodel.add(Dropout(0.4))

# model.add(MaxoutDense(2048,256,nb_feature=8))
# model.add(Dropout(0.4))

revmodel.add( GRU(feat_dims, 512 , return_sequences=True, activation='tanh') )
revmodel.add(Dropout(0.2))

revmodel.add( GRU(512, 512 , return_sequences=True, activation='tanh') )
revmodel.add(Dropout(0.2))

revmodel.add(TimeDistributedDense(512,2048,activation='relu'))
#revmodel.add(PReLU(2048))
revmodel.add(Dropout(0.4))

revmodel.add(TimeDistributedDense(2048,1024,activation='relu'))
#revmodel.add(PReLU(1024))
revmodel.add(Dropout(0.4))

revmodel.add(TimeDistributedDense(1024,48))
revmodel.add(Activation('time_distributed_softmax'))

revtrainer = Adadelta(lr = 1.0 , rho = 0.97 , epsilon = 1e-8 )
revmodel.compile(loss = 'categorical_crossentropy' , optimizer = revtrainer)

if os.path.exists(model_name):
  model.load_weights(model_name)

if os.path.exists('reverse'+model_name):
  revmodel.load_weights('reverse'+model_name)


for r_n in repeat_num:
  print('trained by frame length:' , r_n)
  X , x_f ,x_p         = fold( X , r_n[0] , feat_dims )
  X_cv , xcv_f , xcv_p = fold( X_cv , r_n[0] , feat_dims )
  Y , y_f , y_p        = fold( Y , r_n[0] , 48)
  Y_cv , ycv_f , ycv_p = fold( Y_cv , r_n[0] , 48)

  revX = np.fliplr(X)
  revX_cv = np.fliplr(X_cv)
  revY = np.fliplr(Y)
  revY_cv = np.fliplr(Y_cv)

  try:
    for i in range(r_n[2]):
      model.fit(X,Y , batch_size =r_n[1],nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
      model.evaluate(X_cv,Y_cv , show_accuracy=True)

      revmodel.fit(revX,revY , batch_size=r_n[1],nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
      revmodel.evaluate(revX_cv,revY_cv , show_accuracy=True)
      
      ans = fold_back(Y_cv , r_n[0] , 48 , ycv_f , ycv_p)
      ans = np.argmax(ans, axis=1)

      pred = ( model.predict(X_cv,batch_size=r_n[1]) + np.fliplr(revmodel.predict(revX_cv,batch_size=r_n[1])) ) / 2
      pred = fold_back(pred , r_n[0] , 48 , xcv_f , xcv_p)
      pred = np.argmax(pred,axis=1)

      correct = 0
      for p,a in zip(pred,ans):
        if p == a:
          correct += 1
      print( 'accuracy: {}'.format(float(correct)/len(pred)) )

  except KeyboardInterrupt:
    print('Stop')
  
  X    = fold_back(X , r_n[0] , feat_dims , x_f , x_p)
  X_cv = fold_back(X_cv , r_n[0] , feat_dims , xcv_f , xcv_p)
  Y    = fold_back(Y , r_n[0] , 48 , y_f , y_p)
  Y_cv = fold_back(Y_cv , r_n[0] , 48 , ycv_f , ycv_p)
  





model.save_weights(model_name)
revmodel.save_weights('reverse'+model_name)

del X
del Y
"""
combine_model = Sequential()

combine_model.add(Dense(96,512,activation='linear'))
combine_model.add(PReLU(512))
combine_model.add(Dropout(0.3))

combine_model.add(Dense(512,256,activation='linear'))
combine_model.add(PReLU(256))
combine_model.add(Dropout(0.3))

combine_model.add(Dense(256,48))
combine_model.add(Activation('softmax'))

trainer = Adadelta(lr = 2.0 , rho = 0.97 , epsilon = 1e-8 )
combine_model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)

prob_cv_1 = getResultByDiffSeq( model , X_cv , repeat_num[3:] , feat_dims , reverse=False)
prob_cv_2 = getResultByDiffSeq( revmodel , X_cv , repeat_num[3:] , feat_dims , reverse=True )

X = np.concatenate( (prob_cv_1,prob_cv_2) , axis=1 )
Y = Y_cv


try:
  combine_model.fit( X , Y , batch_size = 32 , nb_epoch=30 , shuffle=True , validation_split=0.1 , show_accuracy=True)
except KeyboardInterrupt:
  print('Stop')

del revX_cv
del revY_cv
del prob_cv_1
del prob_cv_2
del X
del Y



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

prob_te_1 = getResultByDiffSeq( model , X_test , repeat_num[4:] , feat_dims , reverse=False)
prob_te_2 = getResultByDiffSeq( revmodel , X_test , repeat_num[4:] , feat_dims , reverse=True )

X_te = np.concatenate( (prob_te_1,prob_te_2) , axis=1 )

ans2 = combine_model.predict_classes(X_te)

f_pred = open( 'LSTM_modified_'+pred_filename , 'w')
w = csv.writer(f_pred)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans2):
  predictions.append( [n , dict_48_39[ label_set[int(a)] ]] )
w.writerows(predictions)


f_pred_48 = open('48_LSTM__modified_'+pred_filename , 'w')
w = csv.writer(f_pred_48)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans2):
  predictions.append( [n , label_set[int(a)]] )
w.writerows(predictions)



"""