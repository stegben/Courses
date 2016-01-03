import sys,os
from collections import Counter

hw2_sub_filename = sys.argv[1]
new_folder_name = sys.argv[2]

os.mkdir(new_folder_name)

f_orig = open(hw2_sub_filename , 'r')
f_test = open('./mfcc/test.ark' , 'r')

smooth = False

test_frame_name = set()
for row in f_test:
  framename = row.rstrip().split()[0]
  filename = framename.split('_')[0] + '_' + framename.split('_')[1]
  test_frame_name.add(filename)
# print(test_frame_name)

f_orig.readline()



def SeqToPhone(frame_seq):
  PhoneSeq = []
  cur = None
  for s in frame_seq:
    if s != cur:
      PhoneSeq.append(s)
      cur = s
  return PhoneSeq

def SmoothSeq(frame_seq,window = 3):
  new_seq = []
  for i in range(len(frame_seq)-window+1):
    tmp = frame_seq[i:i+window]
    x = Counter(tmp).most_common(1)[0]
    
    if x[1] >= 2:
      new_seq.append(x[0])
    
    # new_seq.append(x[0])
  return new_seq

cur_inst = None
phone_seq = []
w = []
whole_seq = []
name_seq = []
for row in f_orig:
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
      """
      if new_seq[0] == 'L':
	      del new_seq[0]
	    if new_seq[-1] == 'L':
	      del new_seq[-1]
      """
      # print("".join(new_seq))
      w.append([cur_inst , " ".join(new_seq)])
    phone_seq = []
    phone_seq.append(lab[1])
    cur_inst = inst_name
  phone_seq.append(lab[1])
  whole_seq.append(lab[1])

if smooth:
  new_seq = SeqToPhone( SmoothSeq(phone_seq) )
else:
  new_seq = SeqToPhone( phone_seq )
"""
if new_seq[0] == 'L':
  del new_seq[0]
if new_seq[-1] == 'L':
  del new_seq[-1]
"""
  # print("".join(new_seq))
w.append([cur_inst , " ".join(new_seq)])

for ans in w:
  # print(ans[0])
  if ans[0] in test_frame_name:
    f = open(new_folder_name+'/'+ans[0] , 'w')
    f.write(ans[1])
    f.close()