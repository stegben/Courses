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

from Layer import Layer
from MLP import MLP
import MLPtrainer

'''
TODO:
1. drop-out validation
2. k-fold validation
3. store model and value
'''

#########################################################
#
# Initialization
#
#########################################################

###### parameter setting
epoch = 300
batch_size = 100
hiddenStruct = [100 , 15 , 50 ,20]
alpha = 0.2
momentum = 0.9
L1 = 0.0
L2 = 0.0
dropout = 0.1

feat_num = 39

##########################
# normalization
##########################
if os.path.exists('./MFCCNorm.pickle'):
  fnorm = open('MFCCNorm.pickle' , 'r')
  norm , minv = pickle.load(fnorm)
else: 
  ftr = open('./mfcc/train.ark' , 'r') #training set
  d = [[] for x in range(feat_num)]
  for row in ftr:
    row = row.rstrip().split(" ")
    feat = [ float(a) for a in row[1:] ]
    for h in range(feat_num):
    	d[h].append(feat[h])
  norm = [1] * feat_num
  minv = [0] * feat_num
  for ind in range(feat_num):
    norm[ind] = 4 / (max(d[ind]) - min(d[ind]))
    minv[ind] = min(d[ind])
  ftr.close()
  fnorm = open('MFCCNorm.pickle' , 'w')
  pickle.dump((norm,minv) , fnorm)
fnorm.close()

def normalize(x):
  return [(a-m)*b-2 for a,b,m in zip(x,norm,minv)]

####### all files used
ftr = open('./mfcc/train.ark' , 'r') #training set
fte = open('./mfcc/test.ark'  , 'r') # testing set
flab = open('./label/train.lab'  , 'r') # label
fmap = open('./phones/48_39.map' , 'r') # label mapping 48-39

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


#########################################################
#
# Training
#
#########################################################

i = 1
correct = 0
error = []
feat_batch = []
lab_batch = []

##### start training
print('start training by MFCC...')
for k in range(epoch):
  print("start epoch: %d" % (k+1) )
  
  # iteration through all training examples
  for row in ftr :
    r = random.random()
    if r > dropout:
      continue
    # feature vector
    row = row.rstrip().split(" ")
    feat = normalize([ float(a) for a in row[1:] ])
    feat_batch.append( feat )
    
    # label vector
    temp = [0] * 48 
    temp[ labelSet.index(label_dict[row[0]]) ] = 1
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
      # get a predition
      p = pred(feat)
      largestInd = max( (v, i) for i, v in enumerate(p[0]) )[1]
      # print(labelSet[largestInd])
      # print(label_dict[row[0]])
      if labelSet[largestInd] == label_dict[row[0]]:
        correct += 1
    
    # show progress
    if i % 10000 == 0:
      print('train instances number: %d' % i)
      print('error: %f' % error[-1])
      print(correct)
      correct = 0

    i += 1
	# end for loop: loop through all file

  ftr.close()
  ftr = open('./fbank/train.ark' , 'r')
  # end for loop: epoch  
plt.plot(error)
plt.show()

#########################################################
#
# Testing
#
#########################################################



###### write in file preparation
fresult = open('result_mfcc.csv' , 'w')
w = csv.writer(fresult)
w.writerow(['Id' , 'Prediction'])

###### start predicting 
print('start predicting...')
for row in fte:
  
  # feature vector
  row = row.rstrip().split(" ")
  input = normalize([ float(a) for a in row[1:] ] )
  input = np.array(input)
  
  # get prediction distribution
  out = pred(input)
  # print(out)

  largestInd = max( (v, i) for i, v in enumerate(out[0]) )[1]
  # print(largestInd)
  label = map_48_39[ labelSet[largestInd] ]
  
  result = []
  result.append(row[0]) # id
  result.append(label) # label
  
  w.writerow(result)


print('predicting done.')
fresult.close()
ftr.close()
fte.close()
flab.close()
fmap.close()