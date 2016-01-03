"""
This file aims to train a RNNLM
"""

import sys,os
import numpy as np

import theano

from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, MaxoutDense,Reshape, TimeDistributedDense
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import GRU, LSTM


LModel_name = sys.argv[1]
tr_filename = sys.argv[2]
# training text file should be one sentence per line
vocab_size = 5000
wordSeqLen = [ None,3,5,None ]
# predict_after = 1




ftr = open(tr_filename , 'r')



# create trining matrix


  
"""
1. count instance number
2. create np array ( instance_number , length , vocab_size)
"""
def WordSeqPreprocess(f , wl):
  start = ['<s>']
  end   = ['</s>']
  X=[]
  Y=[]
  if wl is not None:
    seq_len = wl
    for line in f:
      line = ''.join([c for c in line if c not in ('!','.','?')])
      words = line.split()
      # print(words)
      words = start + words + end
      if len(words) <= wl:
        continue
      for ind in range( len(words)-wl ):
        X.append( words[ ind   : ind+wl   ] )
        Y.append( words[ ind+1 : ind+wl+1 ] )
  else:
    max_len=0
    for line in f:
      line = ''.join([c for c in line if c not in ('!','.','?')])
      if len(line.split()) > max_len:
        max_len = len(line.split())
    seq_len = max_len + 1
    f.seek(0)
    for line in f:
      line = ''.join([c for c in line if c not in ('!','.','?')])
      words = line.split()
      words = start + words + end
      for i in range(max_len + 2 - len(words)):
        words = words + end
      X.append( words[:-1] )
      Y.append( words[1:]  )
  return X,Y,len(X),seq_len

trainDataSets = []
for l in wordSeqLen:
  xWordSeq , yWordSeq , instNum , seqSize = WordSeqPreprocess( ftr , l )
  print(instNum , seqSize)
  X = np.zeros( (instNum , seqSize , vocab_size) , dtype=theano.config.floatX )
  Y = np.zeros( (instNum , seqSize , vocab_size) , dtype=theano.config.floatX )
  for i , (sentX , sentY) in enumerate(zip(xWordSeq , yWordSeq)):
    for s , (wordX , wordY) in enumerate(zip(sentX , sentY)):
      X[i , s , hash(wordX)%vocab_size ] = 1
      Y[i , s , hash(wordY)%vocab_size ] = 1
  ftr.seek(0)
  trainDataSets.append( (X,Y) )


######## create model

model = Sequential()

model.add( GRU(vocab_size, 512 , return_sequences=True, activation='tanh') )
model.add(Dropout(0.3))

model.add( GRU(512, 512 , return_sequences=True, activation='tanh') )
model.add(Dropout(0.3))

model.add(TimeDistributedDense(512,1024,activation='tanh'))
model.add(Dropout(0.5))

model.add(TimeDistributedDense(1024,1024,activation='relu'))
model.add(Dropout(0.4))

model.add(TimeDistributedDense(1024,vocab_size))
model.add(Activation('time_distributed_softmax'))

trainer = Adadelta(lr = 0.07 , rho = 0.97 , epsilon = 1e-8 )
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)

if os.path.exists(LModel_name):
  model.load_weights(LModel_name)



####### train model

for trainData in trainDataSets:
  try:
    model.fit( trainData[0] ,
               trainData[1] , 
               batch_size =32 , 
               nb_epoch=1, 
               shuffle=True , 
               validation_split=0.0,
               show_accuracy=False)
  except KeyboardInterrupt:
    print('Stop')

model.save_weights(LModel_name)