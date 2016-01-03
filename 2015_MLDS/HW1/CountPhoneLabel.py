import matplotlib.pyplot as plt
import csv
from collections import Counter

flab = open('./label/train.lab'  , 'r') # label
fmap = open('./48_idx_chr.map_b' , 'r') # label to 
fte =  open('./old_HW1_best.csv' , 'r')
fsub = open('HW2sub.csv' , 'w')



map_48_chr = {}
for row in fmap:
  l = row.rstrip().split()
  map_48_chr[ l[0] ] = l[2]

def SeqToPhone(frame_seq):
  PhoneSeq = []
  cur = None
  for s in frame_seq:
    if s != cur:
      PhoneSeq += map_48_chr[s]
      cur = s
  return PhoneSeq

def SmoothSeq(frame_seq,window = 7):
  new_seq = []
  for i in range(len(frame_seq)-window+1):
    tmp = frame_seq[i:i+window]
    x = Counter(tmp).most_common(1)[0]
    
    if x[1] >= 4:
      new_seq.append(x[0])
    
    # new_seq.append(x[0])
  return new_seq

st = []
cur = None
num = 5
for row in flab:
  lab = row.rstrip().split(",")
  if lab[1] != cur:
  	st.append(num)
  	cur = lab[1]
  	num = 1
  else:
  	num += 1
'''
plt.hist(st , bins = range(30))
plt.show()
'''

cur_inst = None
phone_seq = []
w = csv.writer(fsub)
w.writerow(['id' , 'phone_sequence'])
fte.readline()
for row in fte:
  lab = row.rstrip().split(",")
  inst_name = lab[0].split("_")
  inst_name = inst_name[0] + "_" + inst_name[1]
  if inst_name != cur_inst :
  	if cur_inst:
  	  new_seq = SeqToPhone(SmoothSeq(phone_seq))
  	  if new_seq[0] == 'L':
  	  	del new_seq[0]
  	  if new_seq[-1] == 'L':
  	    del new_seq[-1]
  	  # print("".join(new_seq))
  	  w.writerow([cur_inst , "".join(new_seq)])
  	phone_seq = []
  	phone_seq.append(lab[1])
  	cur_inst = inst_name
  phone_seq.append(lab[1])

new_seq = SeqToPhone(SmoothSeq(phone_seq))
if new_seq[0] == 'L':
  del new_seq[0]
if new_seq[-1] == 'L':
  del new_seq[-1]
  # print("".join(new_seq))
w.writerow([cur_inst , "".join(new_seq)])

