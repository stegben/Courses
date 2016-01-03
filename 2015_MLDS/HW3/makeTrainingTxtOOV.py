import pdb
import re

f = open('tr2.txt','r')
paragraph = f.read()
f.close()
words = paragraph.split()
voc = {}
for w in words:
	if w in voc:
		voc[w] = voc[w] + 1
	else:
		voc[w] = 1

voc['<OOV>'] = 0
OOVs = []
for w in voc.keys():
	if voc[w] < 5:
		voc.pop(w)
		voc['<OOV>'] = voc['<OOV>'] + 1
		OOVs.append(w)

replace_dic = {}
for w in OOVs:
	replace_dic[w] = '<OOV>'

sentences = paragraph.split('\n')
sent_word = []
for se in sentences:
	s = se.split()
	sent_word.append(s)
#print "Start replacing OOV"
for i,sen in enumerate(sent_word):
	for j,w in enumerate(sen):
		if w in replace_dic:
			sent_word[i][j] = '<OOV>'
print(len(sent_word))
print(sent_word[1])

"""
for sen in sent_word:
	for w in sen:
		print w,
	print '\n',
"""