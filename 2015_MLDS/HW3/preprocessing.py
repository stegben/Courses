
import string, sys, os
import numpy as np
import re

def getWordDict(filename):
	f = open(filename , 'r') # test file
	regex = re.compile('^(\d+).\) (.+) \[(.+)\] (.+)')

	word_dict = {}
	word_hash = 0
	for row in f:
		match = regex.findall(row.lower())
		match = list(match[0])
		word_list = match[1].split() + [match[2]] + match[3].split()
		for w in word_list:
			if w not in word_dict.keys():
				word_hash += 1
				word_dict[w] = word_hash
	return word_dict,word_hash	

def getTest(filename , word_dict):
	"""
	return a list:
	{
		key: number
		value:{ [pre] , [post] , [choices]}
		...
	}
	"""
	f = open(filename , 'r')
	regex = re.compile('^(\d+).\) (.+) \[(.+)\] (.+)')

	test_set = {}
	tmp = {}
	cur = 0
	for row in f:
		match = regex.findall(row.lower())
		match = list(match[0])
		if int(match[0]) == cur:
			# tmp['choice'].append(word_dict[match[2]])
			tmp['choice'].append(match[2])
			continue
		else:
			test_set[cur] = tmp
			cur = int(match[0])
			tmp = {}
			tmp['pre'] = match[1].split()
			# tmp['pre'] = [word_dict[w] for w in match[1].split()]
			tmp['post'] = match[3].split()
			# tmp['post'] = [word_dict[w] for w in match[3].split()]
			tmp['choice'] = [match[2]]
			# tmp['choice'] = [word_dict[match[2]]]

	test_set[cur] = tmp
	return test_set

def getTrain(train_folder , word_dict):
	tr = []
	for f in os.listdir(train_folder):
		f = open(train_folder+'/'+f , 'r')
		al = ' '.join(line.rstrip('\n\r').lower() for line in f)
		# re.sub('\[[^\[^\]]*\]' , '',al)

		al = re.sub('\[[^\[^\]]*\]|\*+[^\*^\r^\n]+\*+|<+[^<^>]+>+|\^\d+|' , '' , al)
		al = re.split('\W+',al)
		tr = tr+al

	# tr = [word_dict.get(key,0) for key in tr]
	return tr

def skipgrams(sequence, vocabulary_size, 
    window_size=4, negative_samples=1., shuffle=True, 
    categorical=False, sampling_table=None):
    ''' 
        Take a sequence (list of indexes of words), 
        returns couples of [word_index, other_word index] and labels (1s or 0s),
        where label = 1 if 'other_word' belongs to the context of 'word',
        and label=0 if 'other_word' is ramdomly sampled

        @param vocabulary_size: int. maximum possible word index + 1
        @param window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
        @param negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
        @param categorical: bool. if False, labels will be integers (eg. [0, 1, 1 .. ]), 
            if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]

        Note: by convention, index 0 in the vocabulary is a non-word and will be skipped.
    '''
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[i] < random.random():
                continue

        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[words[i%len(words)], random.randint(1, vocabulary_size-1)] for i in range(nb_negative_samples)]
        if categorical:
            labels += [[1,0]]*nb_negative_samples
        else:
            labels += [0]*nb_negative_samples

    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels


def getWord2VecDict(filename):
	f = open(filename , 'rb')
	w2v_dict = {}
	f.readline()
	for line in f:
		line = line.rstrip().split()
		w2v_dict[line[0]] = [float(w) for w in line[1:]]
	return w2v_dict