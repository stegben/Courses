
import sys,os
import re
import numpy as np
import theano

from preprocessing import getWordDict, getTest, getTrain, skipgrams, getWord2VecDict

from model.layers.embeddings import Embedding
from model.models import Sequential
from model.optimizers import Adadelta
from model.layers.core import Dense , PReLU , Dropout , Activation
from model.layers.recurrent import GRU

s = 200 # split training set
sen = 150 # sentence length
epochs = 1



train_filename = sys.argv[1]
test_filename = sys.argv[2]
wv_filename = sys.argv[3]
output_filename = sys.argv[4]

"""
train_filename = 'tr.txt'
test_filename = 'testing_data.txt'
wv_filename = './word2vec/vectors.bin'
output_filename = 'output.txt'
"""
print("...get word dictionary from testing data")
word_dict,word_num = getWordDict(test_filename) 

test = getTest(test_filename , word_dict)

print("...get training words")
ftr = open(train_filename , 'r')
train_all = ftr.readline().rstrip().split()
# train_all = getTrain(train_folder , word_dict)

train = train_all[:-1]
label = train_all[1:]

# train = train[:100000]
# label = label[:100000]

del train_all # release memory
label = [word_dict.get(word,0) for word in label]
"""
new_label = []
for word in label[:8000000]:
	l = [0]*(word_num+1)
	l[word_dict.get(word,0)] = 1
	new_label.append([l])
label = new_label
del new_label
"""
print("...get word2vec dictionary")
word2vec_dict = getWord2VecDict(wv_filename)
train = [word2vec_dict.get(word,[0]*200) for word in train]

train = zip(*[iter(train)]*sen)
label = zip(*[iter(label)]*sen)

"""
print("...get train and label")
train_vec = np.array([[word2vec_dict.get(word,[0]*200)] for word in train] , dtype = theano.config.floatX)
label_vec = np.array([[word2vec_dict.get(word,[0]*200)] for word in label] , dtype = theano.config.floatX)
# label_vec = np.array([word2vec_dict[word] for word in label] , dtype = theano.config.floatX)
"""

print("...create model")
model = Sequential()

model.add( GRU(200 , 100 , return_sequences=True) )
model.add( Dropout(0.1) )

model.add( Dense(100, (word_num+1) , init="he_normal" ) )
model.add(Activation('time_distributed_softmax'))
"""
model.add( GRU(200 , 200 , return_sequences=True) )
model.add( Dropout(0.5) )

model.add( Dense(200,512 , init="he_normal" , activation="linear") )
model.add( PReLU(512) )
model.add( Dropout(0.2) )

model.add( Dense(512, (word_num+1) , init="he_normal" ) )
model.add(Activation('time_distributed_softmax'))

"""

trainer = Adadelta( lr = 0.2 , rho = 0.95)
model.compile( loss = "categorical_crossentropy" , optimizer = trainer)


train_num = len(train)

for epoch in range(epochs):
	print('_______in epoch: ',epoch)
	for split in range(s):
		print('_______in split: ',split)
		l = [0]*(word_num+1)
		train_vec = np.array(train[(train_num/s * split) : (train_num/s * (split+1))] , dtype = theano.config.floatX)
		label_vec = []
		for w_v in label[(train_num/s * split) : (train_num/s * (split+1))]:
			tmp = []
			for w in w_v:
				l[w] = 1
				tmp.append(l)
				l = [0]*(word_num+1)
			label_vec.append(tmp)
		label_vec = np.array(label_vec, dtype = theano.config.floatX)
		model.fit(train_vec , label_vec , batch_size = 1 , nb_epoch = 1 , validation_split=0.0 , shuffle=False)

model.save_weights('model.ke')


answer_dict = {0:'a',1:'b',2:'c',3:'d',4:'e'}
fo = open(output_filename , 'w')
fo.write("Id,Answer\n")
for key in sorted(test):
	if key is 0:
		continue
	print(key)
	i = [[word2vec_dict.get(p,[0]*200) for p in test[key]['pre']]]
	result = model.predict(np.array(i , dtype = theano.config.floatX) , batch_size = 1)
	ch = [result[0][-1][word_dict[c]] for c in test[key]['choice']]
	ans = answer_dict[ch.index(max(ch))]
	fo.write("%s\n" % ','.join([str(key),ans]))


fo.close()

# print(train[:3000])

# word_dict.get(key,0)
