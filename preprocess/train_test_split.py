from collections import Counter
import os
from shutil import copyfile
import sys

from sklearn.cross_validation import train_test_split

def createXY(dirName, exSet):
	listFile = [f for f in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, f))]
	X, y = [], []
	for f in listFile:
		idx = f.find('#')
		if idx != -1:
			label, fName = f[0:idx], f[idx+1:]
			if label in exSet:
				continue
			y.append(label)
			X.append(fName)
	print dirName, len(y)
	c = Counter(y)
	for x in sorted(c.keys()):
		print '\t', x, c[x]
	return X, y

def main():
	# TEXTDIR, TESTDIR, TRAINDIR = "raw_text", "test", "train"
	TEXTDIR, TESTDIR, TRAINDIR = sys.argv[1], sys.argv[3], sys.argv[2]
	test_size = float(sys.argv[4])
	exList = []
	if len(sys.argv) > 5:
		exList.extend(sys.argv[5:])

	print 'Loading all data', TEXTDIR
	X, y = createXY(TEXTDIR, set(exList))
	# print Counter(y)

	print 'Splitting data', 1 - test_size, '-', test_size
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=43, stratify=y)

	print 'Generating train set', TRAINDIR, len(y_train)
	for i in range(len(y_train)):
		fileName = os.path.join(TEXTDIR, y_train[i] + '#' + X_train[i])
		newFileName = os.path.join(TRAINDIR, y_train[i] + '#' + str(i) + ".txt")
		copyfile(fileName, newFileName)

	print 'Generating test set', TESTDIR, len(y_test)
	for i in range(len(y_test)):
		fileName = os.path.join(TEXTDIR,  y_test[i] + '#' + X_test[i])
		newFileName = os.path.join(TESTDIR, y_test[i] + '#' + str(i) + ".txt")
		copyfile(fileName, newFileName)

	print ""
if __name__ == '__main__':
	main()