import wave
from os import listdir # get all wav file
import os.path # use os.path.split() to get raw wav name
from struct import unpack
import csv

flab = open('./label/train.lab'  , 'r') # label
label_dict = {}
for row in flab:
  lab = row.rstrip().split(",")
  label_dict[lab[0]] = lab[1] 
labelSet  = [ 'aa','ae', 'ah','ao', 'aw','ax','ay', 'b',
        'ch','cl',  'd','dh', 'dx','eh','el','en',
       'epi','er', 'ey', 'f',  'g','hh','ih','ix',
        'iy','jh',  'k', 'l',  'm','ng', 'n','ow',
              'oy', 'p',  'r','sh','sil', 's','th','t',
        'uh','uw','vcl', 'v',  'w', 'y','zh', 'z']

def yieldWav():
  for wavName in listdir('./wav'):
    fwav = wave.open('./wav/' + wavName , 'r')
    nframes = fwav.getnframes()
    data  = unpack( "%ih"%nframes , fwav.readframes(nframes) )
    data = [ float(i) / (2**13) for i in data ]
    fwav.close()
    wavName = os.path.splitext(wavName)[0]
    yield wavName , data

def yieldFrame( data , width = int(0.025*16000) , step = int(0.01*16000) ):
  nChunks = ( (len(data)-width)/step ) + 1
  for i in range( 0 , nChunks * step , step ) :
    if i + width > len(data): 
      break
    feat = data[ i : i+width ]
    yield feat


wavTrainFile = open('train_wav.csv' , 'w')
wavTestFile  = open('test_wav.csv' , 'w')
w_tr = csv.writer(wavTrainFile)
w_te = csv.writer(wavTestFile)

for a,data in yieldWav():
  if a + '_1' in label_dict :
    for ind , feat in enumerate(yieldFrame(data)) :
      name = a + '_' + str(ind+1)
      feat = [ labelSet.index(label_dict[name]) ] + feat
      w_tr.writerow(feat)
  else :
    for ind , feat in enumerate(yieldFrame(data)) :
      name = a + '_' + str(ind+1)
      feat = [name] + feat
      w_te.writerow(feat)

wavTrainFile.close()
wavTestFile.close()