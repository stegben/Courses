import sys, os
import csv
from collections import Counter


frame_csv_filename = sys.argv[1]
output_filename = sys.argv[2]
smooth = True

fte = open(frame_csv_filename,'r')
fmap = open('./conf/48_idx_chr.map_b' , 'r')
fsub = open(output_filename,'w')

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

def SmoothSeq(frame_seq,window = 4):
  new_seq = []
  for i in range(len(frame_seq)-window+1):
    tmp = frame_seq[i:i+window]
    x = Counter(tmp).most_common(1)[0]
    
    if x[1] >= 3:
      new_seq.append(x[0])
    
    # new_seq.append(x[0])
  return new_seq

cur_inst = None
phone_seq = []
w = csv.writer(fsub)
w.writerow(['id' , 'phone_sequence'])
fte.readline()
whole_seq = []
name_seq = []
for row in fte:
  lab = row.rstrip().split(",")
  name_seq.append(lab[0])
  inst_name = lab[0].split("_")
  inst_name = inst_name[0] + "_" + inst_name[1]
  if inst_name != cur_inst :
    if cur_inst:
      if smooth:
        new_seq = SeqToPhone( SmoothSeq(phone_seq) )
      else:
        new_seq = SeqToPhone( phone_seq )
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
  whole_seq.append(lab[1])

if smooth:
  new_seq = SeqToPhone( SmoothSeq(phone_seq) )
else:
  new_seq = SeqToPhone( phone_seq )
if new_seq[0] == 'L':
  del new_seq[0]
if new_seq[-1] == 'L':
  del new_seq[-1]
  # print("".join(new_seq))
w.writerow([cur_inst , "".join(new_seq)])
