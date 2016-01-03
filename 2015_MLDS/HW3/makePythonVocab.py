import sys
import numpy as np
import math
import struct as st
import cPickle

fDump = open('vector_pyth.pkl',"wb")

f = open(sys.argv[1],mode="rb")
a = f.read()
# read # of words and size of vectors
# for progressbar
toolbar_width = 100
vocab = {}
# 
i = 0
pi = 0
wordNum = 0
sizeVec = 0
while a[i]!=' ':
	i = i + 1
wordNum = int(a[0:i])
a = a[i:]
i = 0
while a[i]!='\n':
	i = i + 1
sizeVec = int(a[0:i])
a = a[i:]
print "words: %d size: %d" % (wordNum, sizeVec)

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
tmp_idx = -1

for i in range(0,wordNum):
	idex = a.find(' ',pi)
	word = a[pi:idex]
	word = word.lstrip()
	pi = idex+1
	wvec = st.unpack("f"*sizeVec, a[pi:pi+sizeVec*4])
	wvec = np.array(wvec)
	pi = pi+sizeVec*4
	vocab[word] = wvec
	if( math.floor(i*toolbar_width/wordNum) != tmp_idx):
		tmp_idx = tmp_idx + 1
		sys.stdout.write("-")
		sys.stdout.flush()
sys.stdout.write("\n")

cPickle.dump(vocab, fDump)
f.close()
fDump.close()