import sys
import editdistance

ref = open("test.txt",'r')
rec = open(sys.argv[1],'r')

fchmap = open('./conf/timit.chmap','r')
e2c = {}
for line in fchmap:
  e2c[ line.split('\t')[0] ] =  line.split('\t')[1].rstrip()

def eng2ch(x):
  if x in e2c:
    return e2c[x]
  else:
    return ''


ans = {}
score = []
rec.readline()
for line in ref:
	line = line.rstrip()
	line = ''.join([c for c in line if c not in ('!','.','?')])
	words = ' '.join(line.split(',')[1:])
	words = [eng2ch(w.lower()) for w in words.split()]
	ans[ line.split(',')[0] ] = words
for line in rec:
	x1 = [line.rstrip().split(',')[1][i:i+3] for i in range(0, len(line.rstrip().split(',')[1]) , 3) ]
	x2 = ans[line.split(',')[0]] 
	# print(x1)
	# print(x2)
	y = editdistance.eval(x1,x2)
	score.append(y)
print float(sum(score)) / len(score)
