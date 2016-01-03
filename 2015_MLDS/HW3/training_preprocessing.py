import sys,os
import re
"""
create a preprocessed text file
each line is a sentence from 
"""

training_foldername = sys.argv[1]

new_filename = sys.argv[2]



f_new = open( new_filename , 'w' )

for f in os.listdir(training_foldername):
	f = open(training_foldername+'/'+f , 'r')
	al = ' '.join(line.rstrip('\n\r').lower() for line in f)
	# re.sub('\[[^\[^\]]*\]' , '',al)

	al = re.sub('\[[^\[^\]]*\]|\*+[^\*^\r^\n]+\*+|<+[^<^>]+>+|\^\d+|' , '' , al)
	al = re.split('\W+',al)
	f_new.write(" ".join(al))
	"""
	for l in al:
		l = l.lstrip()
		if ( l.startswith("#") or
			 l.startswith("[") or
			 l.startswith("*") or
			 l.startswith("(") or
			 l.startswith(")")  ):
			continue
		f_new.write(l+'\n')
	"""