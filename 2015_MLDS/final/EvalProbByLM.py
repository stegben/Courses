import sys,os
import numpy as np
from time import time

import theano

from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, MaxoutDense,Reshape, TimeDistributedDense
from keras.layers.advanced_activations import PReLU
from keras.layers.recurrent import GRU, LSTM

LModel_name = sys.argv[1]
nbest_test_filename = sys.argv[2]
assert os.path.exists(LModel_name)
output_filename = sys.argv[3]

vocab_size = 5000


fi = open(nbest_test_filename , 'r')
fo = open(output_filename , 'w')

fchmap = open('./conf/timit.chmap','r')




"""
Get Model
"""
model = Sequential()

model.add( GRU(vocab_size, 512 , return_sequences=True, activation='tanh') )
model.add(Dropout(0.2))

model.add( GRU(512, 512 , return_sequences=True, activation='tanh') )
model.add(Dropout(0.2))

model.add(TimeDistributedDense(512,1024,activation='tanh'))
model.add(Dropout(0.4))

model.add(TimeDistributedDense(1024,1024,activation='relu'))
model.add(Dropout(0.4))

model.add(TimeDistributedDense(1024,vocab_size))
model.add(Activation('time_distributed_softmax'))

trainer = Adadelta(lr = 0.5 , rho = 0.97 , epsilon = 1e-8 )
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)

model.load_weights(LModel_name)

"""
loop through all candidates
"""


def WordSeqPreprocess(sentences):
  start = ['<s>']
  end   = ['</s>']
  X=[]
  pad = []
  max_len=0
  for line in sentences:
    if len(line) > max_len:
      max_len = len(line)
  seq_len = max_len + 2
  for line in sentences:
    words = start + line + end
    pad.append(seq_len - len(words))
    for i in range(seq_len - len(words)):
      words = words + end
    X.append( words )
  return X,len(X),seq_len,pad



def getAnswer(X , model):
  """
  X is a list of lists of words
  Pad all of the sentences to same length by </s>
  """
  new_sentences , candidate_num , sent_len , pad = WordSeqPreprocess( X )
  cur_word_vec = np.zeros( (candidate_num , sent_len , vocab_size) , dtype = theano.config.floatX )
  next_word_index = np.zeros( (candidate_num , sent_len) , dtype = theano.config.floatX )
  for i , sent in enumerate(new_sentences):
    for j , word in enumerate(sent):
      ind = hash(word)%vocab_size
      cur_word_vec[i,j, ind] = 1
      next_word_index[i,j] = ind
  pred = model.predict(cur_word_vec , batch_size=128 , verbose=0)
  pred = np.log(pred)
  # score = pred[range(pred.shape[0]) , range(pred.shape[1]) ,]
  score = np.zeros((candidate_num,sent_len))
  for i in range(candidate_num):
    for j in range(sent_len):
      score[i,j] = pred[i,j,next_word_index[i,j]]
  # print(score)

  score = np.sum(score,axis=1)
  # print(len(score))
  best = score.argmax()
  # print(best)
  # print(score[best])
  # print(X[best])
  return X[best]
  
    




curInst = None
curCandi = []
result = []
t1 = time()
for line in fi:
  words = line.rstrip().split()
  if len(words) == 0:
    continue
  # deal with the initial condition
  if not curInst:
    curInst = words[0]
    curCandi.append(words[1:])
    continue
  # if see a new wav file name, then start to predict the current candidates
  if words[0] != curInst:
    print(curInst)
    ans = getAnswer( curCandi , model )
    print(ans)
    result.append( (curInst , ans) )
    curCandi = []
  curInst = words[0]
  curCandi.append( words[1:] )

# last item
ans = getAnswer( curCandi , model )
print(ans)
result.append( (curInst , ans) )

t2 = time()
print('prediction time:')
print(t2-t1)

"""
change to Chinese
Write file 
"""
e2c = {}
for line in fchmap:
  e2c[ line.split('\t')[0] ] =  line.split('\t')[1].rstrip()

def eng2ch(x):
  if x in e2c:
    return e2c[x]
  else:
    return ''
fo.write('id,sequence\n')
for ans in result:
  fo.write(ans[0]+',')
  for w in ans[1]:
    fo.write( eng2ch(w) )
  fo.write('\n')  
fo.close()
fi.close()
fchmap.close()