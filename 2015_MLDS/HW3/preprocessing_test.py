import sys
import re
print sys.argv
f = open(sys.argv[1])
index = 0
for line in f:
	ll = line.split(')')
	sentence = ll[1].lstrip()
	choice = ll[0][-1]
	num = int(ll[0][0:-1])
	m = re.search(r"\[([A-Za-z-',0-9]+)\]", sentence)
	sentence = re.sub(r"\[([A-Za-z]+)\]", '`',sentence)
	m = m.group(0)
	m = re.sub(r"[\[\]]",'',m)
	if index != num:
		print num
		index = num
	print choice+','+m+',' + sentence
f.close()