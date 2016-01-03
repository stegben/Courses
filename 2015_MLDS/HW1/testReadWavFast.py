import wave
from os import listdir # get all wav file
import os.path # use os.path.split() to get raw wav name
from struct import unpack
import numpy as np
import itertools as itr
import theano
import theano.tensor as T
from time import time
import matplotlib.pyplot as plt
import csv
import cPickle as pickle
import random

from Layer import Layer
from MLP import MLP
import MLPtrainer

'''
TODO:
1. drop-out validation
2. k-fold validation
3. store model and value
'''

###################################################################################################
#
# Initialization
#
###################################################################################################

###### parameter setting
pepoch = 1
epoch = 7000
batch_size = 100
hiddenStruct = [600,600,400]
alpha = 0.5
momentum = 0.98
L1 = 0.01
L2 = 0.0
dropout = 0.15
outfilename = 'result_wav.csv'
feat_num = 400

######## model initialization
model = MLP(
	        n_in=feat_num ,
	        n_out=48 ,
	        hidStruct = hiddenStruct
	       )

###### prediction function (theano type)
x = T.vector('x')
pred = theano.function([x] , model.predict(x))

######## model trainer initialization 
trainer = MLPtrainer.MLPtrainer(
	       net = model ,
	       learning_rate = alpha ,
	       momentum = momentum ,
	       L1 = L1 , 
	       L2 = L2 )

ftr  = open('./train_wav.csv' , 'r')
fte  = open('./test_wav.csv' , 'r')
flab = open('./label/train.lab'  , 'r') # label
fmap = open('./phones/48_39.map' , 'r') # label mapping 48-39

# label initialization
labelSet  = [ 'aa','ae', 'ah','ao', 'aw','ax','ay', 'b',
	      'ch','cl',  'd','dh', 'dx','eh','el','en',
	     'epi','er', 'ey', 'f',  'g','hh','ih','ix',
	      'iy','jh',  'k', 'l',  'm','ng', 'n','ow',
              'oy', 'p',  'r','sh','sil', 's','th','t',
	      'uh','uw','vcl', 'v',  'w', 'y','zh', 'z']

map_48_39 = {}
for row in fmap:
  l = row.rstrip().split("\t")
  map_48_39[ l[0] ] = l[1]

 
###### create label dictionary
label_dict = {}
for row in flab:
  lab = row.rstrip().split(",")
  label_dict[lab[0]] = lab[1] 


###################################################################################################
#
# Training
#
###################################################################################################
"""
i=1
feat_batch = []
print('start pre-training by FBank...')
for k in range(pepoch):
  print("start epoch: %d" % (k+1) )
  # pre-training
  ftr.seek(0)
  for row in ftr :
    # no dropout
    #r = random.random()
    #if r < 0.01:
    #  continue
    # feature vector
    row = row.rstrip().split(",")
    feat = [float(k) for k in row[1:]]
    feat_batch.append( feat )

    # print(feat_batch)
    # print(temp)

    # batch full, train the model!
    # if i % batch_size == batch:
    if len(feat_batch) >= batch_size :
      
      X = np.array(feat_batch , dtype = theano.config.floatX )

      model.dAPreTraining(X)
      
      feat_batch = []
    
    # show progress
    if i % 100000 == 0:
      print('pre-train instances number: %d' % i)
    i += 1
  # end for loop: loop through all file
"""

i = 1
correct = 0
error = []
feat_batch = []
lab_batch = []
ftr.seek(0)

print('start training by raw .wav file...')
t0 = time()
for k in range(epoch):
  print("start epoch: %d" % (k+1) )
  
  # iteration through all wav file
  for row in ftr :
    r = random.random()
    if r > dropout:
      continue
    # feature vector
    row = row.rstrip().split(',')
    feat_batch.append( [float(k) for k in row[1:]] )
  
    # label vector
    temp = [0] * 48 
    temp[ int(row[0]) ] = 1
    lab_batch.append(temp)
    # print(feat_batch)
    # print(temp)

    # batch full, train the model!
    # if i % batch_size == batch:
    if len(lab_batch) >= batch_size :
    
      X = np.array(feat_batch , dtype = theano.config.floatX )
      Y = np.array(lab_batch , dtype = theano.config.floatX )

      e = trainer(X , Y)
      error.append(e)
  
      feat_batch = []
      lab_batch = []
      """
      # get a predition
      p = pred(feat)
      largestInd = max( (v, i) for i, v in enumerate(p[0]) )[1]
      # print(labelSet[largestInd])
      # print(label_dict[row[0]])
      if labelSet[largestInd] == label_dict[row[0]]:
        correct += 1
      """
  
    # show progress
    if i % 10000 == 0:
      print('train instances number: %d' % i)
      print('error: %f' % sum(error[(len(error)-100):]))
      # print(correct)
      # correct = 0
    i += 1
  ftr.seek(0)
  # end for loop: all row
# end for loop: epoch
netFile = open('wav_300_300_300.pickle' , 'w')
pickle.dump(model,netFile)  
t1 = time()
print(t1-t0)

###################################################################################################
#
# Testing
#
###################################################################################################


###### write in file preparation
fresult = open(outfilename , 'w')
w = csv.writer(fresult)
w.writerow(['Id' , 'Prediction'])

###### start predicting 
print('start predicting...')
for row in fte :
  row = row.rstrip().split(',')
  # feature vector
  feat = [float(k) for k in row[1:]]
  input = np.array(feat,dtype = theano.config.floatX)

  # get prediction distribution
  out = pred(input)

  largestInd = max( (v, i) for i, v in enumerate(out[0]) )[1]
  # print(largestInd)
  label = map_48_39[ labelSet[largestInd] ]

  result = []
  result.append( row[0] ) # id
  result.append(label) # label

  w.writerow(result)


print('predicting done.')
fresult.close()

