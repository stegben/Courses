from scipy import spatial

ans = open('Ans','w')
ans.write('Id,Answer\n')
f = open('vector','r')

vector = {}
for line in f:
	temp = line.split()
	vector[temp[0]] = [ float(i) for i in temp[1:] ]
Dim = len(vector['<OOV>'])
test = open('test','r')

testLines = []
lineNum = 0
for line in test:
	testLines.append(line.split())
	lineNum = lineNum + 1

#print testLines[1]

A = []
B = []
C = []
D = []
E = []
other = []
result = [0] * 5
check = [1.0] * 4

n2a = {0:'a',1:'b',2:'c',3:'d',4:'e'}
for i in range(0,lineNum/5) :
	for j in testLines[5*i]:
		if j[0] == '[':
			if j[1:len(j)-1] in vector:
				A = vector[j[1:len(j)-1]]
			else:
				A = vector['<OOV>']
	for j in testLines[5*i+1]:
		if j[0] == '[':
			if j[1:len(j)-1] in vector:
				B = vector[j[1:len(j)-1]]
			else:
				B = vector['<OOV>']
	for j in testLines[5*i+2]:
		if j[0] == '[':
			if j[1:len(j)-1] in vector:
				C = vector[j[1:len(j)-1]]
			else:
				C = vector['<OOV>']
	for j in testLines[5*i+3]:
		if j[0] == '[':
			if j[1:len(j)-1] in vector:
				D = vector[j[1:len(j)-1]]
			else:
				D = vector['<OOV>']
	for j in testLines[5*i+4]:
		if j[0] == '[':
			index = testLines[5*i+4].index(j)
			if j[1:len(j)-1] in vector:
				E = vector[j[1:len(j)-1]]
			else:
				E = vector['<OOV>']


	if index < 2:
		v_1 = [0] * Dim
		check[0] = 0
	else:
		check[0] = 1
		if testLines[5*i][index-2] in vector:
			v_1 = vector[testLines[5*i][index-2]]
		else:
			v_1 = vector['<OOV>']

	if index < 1:
		v_2 = [0] * Dim
		check[1] = 0
	else:
		check[1] = 1
		if testLines[5*i][index-1] in vector:
			v_2 = vector[testLines[5*i][index-1]]
		else:
			v_2 =  vector['<OOV>']

	if (index+1) >= len(testLines[5*i]):
		v_3 = [0] * Dim
		check[2] = 0
	else:
		check[2] = 1
		if testLines[5*i][index+1] in vector:
			v_3 = vector[testLines[5*i][index+1]]
		else:
			v_3 =  vector['<OOV>']

	if (index+2) >= len(testLines[5*i]):
		v_4 = [0] * Dim
		check[3] = 0
	else:
		check[3] = 1
		if testLines[5*i][index+2] in vector:
			v_4 = vector[testLines[5*i][index+2]]
		else:
			v_4 =  vector['<OOV>']

	other1 = [ sum(x) for x in zip(v_1,v_2) ]
	other2 = [ sum(x) for x in zip(v_3,v_4) ]
	other = [ sum(x)/sum(check) for x in zip(other1,other2) ]

	result[0] = 1 - spatial.distance.cosine(other, A)
	result[1] = 1 - spatial.distance.cosine(other, B)
	result[2] = 1 - spatial.distance.cosine(other, C)
	result[3] = 1 - spatial.distance.cosine(other, D)
	result[4] = 1 - spatial.distance.cosine(other, E)
	ans.write(str(i+1))
	ans.write(',')
	ans.write(n2a[result.index(max(result))])
	ans.write('\n')
