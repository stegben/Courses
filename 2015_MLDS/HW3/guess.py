import sys
import cPickle
import numpy as np
fLoad = open(sys.argv[1],"rb")
print "Loading vocabulary file..."
vocab = cPickle.load(fLoad)
print "Finished"
def distance(x,y,voc=vocab):
	if x in voc:
		if y in voc:
			return np.linalg.norm(voc[x]-voc[y])
		else:
			return np.linalg.norm(voc[x])
	else:
		if y in voc:
			return np.linalg.norm(voc[y])
		else:
			return 00

def maxChoice(choices):
	dis = []	
	for c in choices:
		score = 0
		for cd in choices:
			if(c == cd):
				continue
			else:
				score = score + distance(c,cd)
		dis.append(score)
	return min(enumerate(dis), key=(lambda x: x[1]))

ftest = open(sys.argv[2],"r")
fOut = open("guess_min.sub","w")
fOut.write("Id,Answer")
fOut.write('\n')
for line in ftest:
	ll = line.split()
	index = ll[0]
	#ll = ll[1:]
	a = maxChoice(ll[1:])
	fOut.write( "%s,%c" % (index,chr(ord('a')+a[0])) )
	fOut.write('\n')

ftest.close()
fLoad.close()
fOut.close()