#!/usr/bin/env python
# Prevent sklearn from throwing 1000's of warnings using n_jobs = -1
import warnings
warnings.filterwarnings("ignore")


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import sys

def read_corpus(corpus_file):
	ngrams = 2
	documents=[]
	labels=[]
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens=line.strip().split()
			if ngrams == 1:
				documents.append(tokens[3:])
			if ngrams == 2:
				documents.append(zip(tokens[3:], tokens[3:][1:]))
			if ngrams == 3:
				documents.append(zip(tokens[3:], tokens[3:][1:], tokens[3:][2:]))
			if ngrams == 4:
				documents.append(zip(tokens[3:], tokens[3:][1:], tokens[3:][2:], tokens[3:][3:]))
			labels.append(tokens[1])
	return documents, labels

if len(sys.argv)==3:
	Xtrain, Ytrain=read_corpus(sys.argv[1])
	Xtest, Ytest=read_corpus(sys.argv[2])
	vec=TfidfVectorizer(stop_words= 'english',preprocessor=lambda x: x, tokenizer=lambda x: x)
	classifier = Pipeline([('vec',vec),('cls',SVC(kernel='sigmoid',gamma=0.8,C=1.0))])
	classifier.fit(Xtrain,Ytrain)
	Yguess = classifier.predict(Xtest)
	print("Accuracy: {}".format(accuracy_score(Ytest,Yguess)))
	print(classification_report(Ytest, Yguess))

else:
	print("Usage: LFDassignment5_SVM_Olivier <trainset.txt> <testset.txt>")


