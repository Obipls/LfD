#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import sys

def read_corpus(corpus_file):
	documents=[]
	labels=[]
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens=line.strip().split()
			documents.append(tokens[3:])
			labels.append(tokens[1])
	return documents, labels

if len(sys.argv)==3:
	Xtrain, Ytrain=read_corpus(sys.argv[1])
	Xtest, Ytest=read_corpus(sys.argv[2])
	vec=TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
	classifier = Pipeline([('vec',vec),('cls',MultinomialNB(alpha=0.20,fit_prior=False,class_prior=None))])
	classifier.fit(Xtrain,Ytrain)
	Yguess = classifier.predict(Xtest)
	print("Accuracy: {}".format(accuracy_score(Ytest,Yguess)))
	print(classification_report(Ytest, Yguess))

else:
	print("Usage: LFDassignment5_SVM_Olivier trainset.txt testset.txt")

