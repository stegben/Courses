
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
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, MaxoutDense,Reshape, TimeDistributedDense
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import GRU, LSTM

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
rnn_modle_name = 'rnn_1'
rev_rnn_modle_name = 'rnn_2'

pre_frame_num = 8
post_frame_num = 8
pre_m2 = 3
post_m2 = 3
holdout = 0.3
epoch = 15

## ( sequence number , batch size , epoch )
repeat_num = [(2,128,3),(5,64,4),(15,25,5),(30,15,5),(50,10,6)]


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
#################################################################################################
"""
Process X,y for training
"""
#################################################################################################
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

# feat_raw = []
Y = np.zeros((tr_row_num,48) , dtype = theano.config.floatX)
Y_cv = np.zeros((cv_row_num,48) , dtype = theano.config.floatX)
X = np.zeros((tr_row_num , feat_num) , dtype = theano.config.floatX)
X_cv = np.zeros((cv_row_num , feat_num) , dtype = theano.config.floatX)

ind_for_tr = 0
ind_for_cv = 0
for i,row in enumerate(ftr) :
  row = row.rstrip().split(" ")
  feat = [ float(a) for a in row[1:] ]  
  if row[0].split('_')[0] in cv_name_list:
    X_cv[ ind_for_cv , : ] = feat
    Y_cv[ ind_for_cv , dict_48_ind[label_dict[row[0]]] ] = 1
    ind_for_cv += 1
  else:
    X[ ind_for_tr , : ] = feat
    Y[ ind_for_tr , dict_48_ind[label_dict[row[0]]] ] = 1
    ind_for_tr += 1

X = convolvArray(X , pre_frame_num , post_frame_num)
X_cv = convolvArray(X_cv , pre_frame_num , post_frame_num)

ftr.close()

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_cv = scaler.transform(X_cv)

t2 = time()
print('time-windowed frame creation spend: %0.2f s' % (t2-t1))



#################################################################################################
"""
Model 1
DNN Keras Model for direct transform
"""
#################################################################################################

model = Sequential()


model.add(Dense(feat_dims,2048,activation='linear'))
model.add(PReLU(2048))
model.add(Dropout(0.4))

model.add(Dense(2048,2048,activation='linear'))
model.add(PReLU(2048))
model.add(Dropout(0.5))

model.add(Dense(2048,2048,activation='linear'))
model.add(PReLU(2048))
model.add(Dropout(0.4))

model.add(Dense(2048,2048,activation='linear'))
model.add(PReLU(2048))
model.add(Dropout(0.5))

model.add(Dense(2048,2048,activation='linear'))
model.add(PReLU(2048))
model.add(Dropout(0.4))

model.add(Dense(2048,48,activation='softmax'))

trainer = Adadelta(lr = 4.0 , rho = 0.97 , epsilon = 1e-8 )
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)

if os.path.exists(model_name):
  model.load_weights(model_name)

try:
  for i in range(epoch):
    model.fit(X,Y , batch_size = 128,nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
    model.evaluate(X_cv,Y_cv , show_accuracy=True)
except KeyboardInterrupt:
  print('Stop')


model.save_weights(model_name)

del X
del Y




#################################################################################################
"""
use Model 1 to predict a first result
"""
#################################################################################################

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
ans1 = model.predict_classes(X_test)

f_pred = open( pred_filename , 'w')

'''
Predict a 39 phone result
'''
w = csv.writer(f_pred)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans1):
  predictions.append( [n , dict_48_39[ label_set[a] ]] )
w.writerows(predictions)
f_pred.close()

'''
Predict a 48 phone result
'''
f_pred_48 = open('48_'+pred_filename , 'w')
w = csv.writer(f_pred_48)
w.writerow(['Id','Prediction'])
predictions = []
for n,a in itr.izip(name,ans1):
  predictions.append( [n , label_set[a]] )
w.writerows(predictions)
f_pred_48.close()

del ans1

#################################################################################################
"""
Model 2
use cv dataset to train a RNN model, based on the probability provided by Model 1
"""
#################################################################################################

####### prepare X,y for model 2
prob = model.predict(X_cv)

new_X_cv = convolvArray(prob,pre_m2,post_m2) 
del prob
sp = cv_row_num * 5 / 6
new_X_cv = np.split(new_X_cv,[sp])
new_Y_cv = np.split(Y_cv,[sp])



rnnmodel = Sequential()

# rnnmodel.add(TimeDistributedDense(feat_dims,2048,activation='relu'))
#rnnmodel.add(PReLU(2048))
# rnnmodel.add(Dropout(0.4))

rnnmodel.add( LSTM(48*(pre_m2+post_m2+1), 512 , return_sequences=True, activation='tanh') )
rnnmodel.add(Dropout(0.2))

rnnmodel.add( GRU(512, 512 , return_sequences=True, activation='tanh') )
rnnmodel.add(Dropout(0.2))

rnnmodel.add(TimeDistributedDense(512,2048,activation='relu'))
#rnnmodel.add(PReLU(2048))
rnnmodel.add(Dropout(0.4))

rnnmodel.add(TimeDistributedDense(2048,1024,activation='relu'))
#rnnmodel.add(PReLU(1024))
rnnmodel.add(Dropout(0.4))

rnnmodel.add(TimeDistributedDense(1024,48))
rnnmodel.add(Activation('time_distributed_softmax'))

trainer = Adadelta(lr = 2.0 , rho = 0.97 , epsilon = 1e-8 )
rnnmodel.compile(loss = 'categorical_crossentropy' , optimizer = trainer)
if os.path.exists(rnn_modle_name):
  rnnmodel.load_weights(rnn_modle_name)


revmodel = Sequential()

# revmodel.add(TimeDistributedDense(feat_dims,2048,activation='relu'))
#revmodel.add(PReLU(2048))
# revmodel.add(Dropout(0.4))

revmodel.add( LSTM(48*(pre_m2+post_m2+1), 512 , return_sequences=True, activation='tanh') )
revmodel.add(Dropout(0.2))

revmodel.add( GRU(512, 512 , return_sequences=True, activation='tanh') )
revmodel.add(Dropout(0.2))

revmodel.add(TimeDistributedDense(512,2048,activation='relu'))
#revmodel.add(PReLU(1024))
revmodel.add(Dropout(0.4))

revmodel.add(TimeDistributedDense(2048,1024,activation='relu'))
#revmodel.add(PReLU(1024))
revmodel.add(Dropout(0.4))

revmodel.add(TimeDistributedDense(1024,48))
revmodel.add(Activation('time_distributed_softmax'))

# revtrainer = Adadelta(lr = 2.0 , rho = 0.97 , epsilon = 1e-8 )
revmodel.compile(loss = 'categorical_crossentropy' , optimizer = trainer)
if os.path.exists(rev_rnn_modle_name):
  revmodel.load_weights(rev_rnn_modle_name)


########## start training by diferent length

for r_n in repeat_num:
  print('trained by frame length:' , r_n)
  X , x_f ,x_p         = fold( new_X_cv[0] , r_n[0] , 48*(pre_m2+post_m2+1) )
  X_cv , xcv_f , xcv_p = fold( new_X_cv[1] , r_n[0] , 48*(pre_m2+post_m2+1) )
  Y , y_f , y_p        = fold(new_Y_cv[0] , r_n[0] , 48)
  Y_cv , ycv_f , ycv_p = fold(new_Y_cv[1] , r_n[0] , 48)

  revX = np.fliplr(X)
  revX_cv = np.fliplr(X_cv)
  revY = np.fliplr(Y)
  revY_cv = np.fliplr(Y_cv)

  try:
    for i in range(r_n[2]):
      rnnmodel.fit(X,Y , batch_size =r_n[1],nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
      rnnmodel.evaluate(X_cv,Y_cv , show_accuracy=True)

      revmodel.fit(revX,revY , batch_size=r_n[1],nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
      revmodel.evaluate(revX_cv,revY_cv , show_accuracy=True)
      
      ans = fold_back(Y_cv , r_n[0] , 48 , ycv_f , ycv_p)
      ans = np.argmax(ans, axis=1)

      pred = ( rnnmodel.predict(X_cv,batch_size=r_n[1]) + np.fliplr(revmodel.predict(revX_cv,batch_size=r_n[1])) ) / 2
      pred = fold_back(pred , r_n[0] , 48 , xcv_f , xcv_p)
      pred = np.argmax(pred,axis=1)

      correct = 0
      for p,a in zip(pred,ans):
        if p == a:
          correct += 1
      print( 'accuracy: {}'.format(float(correct)/len(pred)) )

  except KeyboardInterrupt:
    print('Stop')
  """ no need to reshape back, since new_X_cv and new_Y_cv are existing
  X    = fold_back(X , r_n[0] , feat_dims , x_f , x_p)
  X_cv = fold_back(X_cv , r_n[0] , feat_dims , xcv_f , xcv_p)
  Y    = fold_back(Y , r_n[0] , 48 , y_f , y_p)
  Y_cv = fold_back(Y_cv , r_n[0] , 48 , ycv_f , ycv_p)
  """
del X
del X_cv
del Y
del Y_cv
del revX
del revX_cv
del revY
del revY_cv

rnnmodel.save_weights(rnn_modle_name)
revmodel.save_weights(rev_rnn_modle_name)
######### a small model to combine the result of RNN and revRNN

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

prob_cv_1 = getResultByDiffSeq( rnnmodel , new_X_cv[1] , repeat_num , 48*(pre_m2+post_m2+1) , reverse=False)
prob_cv_2 = getResultByDiffSeq( revmodel , new_X_cv[1] , repeat_num , 48*(pre_m2+post_m2+1) , reverse=True )

X = np.concatenate( (prob_cv_1,prob_cv_2) , axis=1 )
Y = new_Y_cv[1]
combine_model.fit( X , Y , batch_size = 32 , nb_epoch=10 , shuffle=True , validation_split=0.1 , show_accuracy=True)




#################################################################################################
"""
use Model 2 to get a modified result of test set
"""
#################################################################################################

test_prob = model.predict(X_test)

new_X_test = convolvArray(test_prob,pre_m2,post_m2)

prob_te_1 = getResultByDiffSeq( rnnmodel , new_X_test , repeat_num , 48*(pre_m2+post_m2+1) , reverse=False)
prob_te_2 = getResultByDiffSeq( revmodel , new_X_test , repeat_num , 48*(pre_m2+post_m2+1) , reverse=True )

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