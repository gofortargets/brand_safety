import cPickle as pkl
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import scipy

from knx.util.logging import Timing

LABEL_SEPARATOR = '#'
STR_FORMAT = "%.4f, %.4f, %.4f, "

outputdir = sys.argv[1]
output_train = os.path.join(outputdir, "results.csv")    
fo = open(output_train, "w")

def predict(clf, doc_term):
	scores = clf.decision_function(doc_term)
	scores[scores >= 0.0] = 1
	scores[scores < 0.0] = 0
	return scores

def convertY(Y, label_to_idx):
	nb_classes = len(label_to_idx)
	Y_new = np.zeros((len(Y), nb_classes))
	for k in range(len(Y_new)):
		if type(Y[k]) == str:
			idx = label_to_idx[Y[k]]
		else:
			idx = Y[k]
		Y_new[k][idx] = 1
	return Y_new

def convert_to_idx(Y, label_to_idx):
	Y_new = [label_to_idx[x] for x in Y]
	return Y_new

def train_test(X_train, y_train, X_test, y_test, filename_ls, C=1):
	#clf = svm.LinearSVC(C=C)
	clf = OneVsRestClassifier(svm.LinearSVC(C=C, class_weight='auto'), n_jobs=-2)
	with Timing("Training with C = " + str(C) + " ..."):
		clf.fit(X_train, y_train)
		joblib.dump(clf, os.path.join(sys.argv[1], "svm_model.pkl"))

	with Timing("Evaluating model ..."):
		# preds = predict(clf, X_test)
		preds = clf.predict(X_test)
		acc = accuracy_score(y_test, preds)
		score_str = str(C) + ", %.4f, "%acc
		f1 = f1_score(y_test, preds, pos_label=None, average='weighted')
		precision = precision_score(y_test, preds, pos_label=None, average='weighted')
		recall = recall_score(y_test, preds, pos_label=None, average='weighted')
		weighted_str = STR_FORMAT%(f1, precision, recall)

	print "Accuracy: " + str(acc)
	print "F1: " + str(f1)
	print "Precision" + str(precision)
	print "Recall" + str(recall)

	print ''
	print 'For each category'
	print metrics.classification_report(y_test, preds)
	cnf_matrix = confusion_matrix(y_test, preds)
	classes = ['Adult', 'Car_accident', 'Death_tragedy', 'Hate_speech', 'Religion', 'Safe']
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=classes,
						  title='Confusion matrix, without normalization')
	plt.savefig('cnf_matrix_normalize.png')
	for i, e in enumerate(y_test):
		if e == 'Hate_speech' and preds[i] != 'Hate_speech':
			print preds[i], filename_ls[i]

def main():
	outputdir = sys.argv[1]
	output_train = os.path.join(outputdir, "train.pkl")
	with Timing("Loading " + output_train + "..."):
		with open(output_train, "rb") as fp:
			X_train = pkl.load(fp)
			y_train = pkl.load(fp)
			train_classes = pkl.load(fp)
			vocabulary = pkl.load(fp)
			mapping = pkl.load(fp)

	output_train = os.path.join(outputdir, "test.pkl")
	with Timing("Loading " + output_train + "..."):
		with open(output_train, "rb") as fp:
			X_test = pkl.load(fp)
			y_test = pkl.load(fp)
			filename_ls = pkl.load(fp)

	#C_list = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 125]
	C_list = [ 1 ]
	#C_list = [1]
	for x in C_list:
		train_test(X_train, y_train, X_test, y_test, filename_ls, C=x)
	fo.close()


if __name__ == '__main__':
	main()
