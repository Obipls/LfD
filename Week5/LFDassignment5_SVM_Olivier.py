#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem.wordnet import WordNetLemmatizer
import sys

def read_corpus(corpus_file):
	wnl = WordNetLemmatizer()
	ngrams = 2
	documents=[]
	labels=[]
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens=line.strip().split()
			lemmedTokens = [wnl.lemmatize(token) for token in tokens[3:]]
			if ngrams == 1:
				documents.append(lemmedTokens)
			if ngrams == 2:
				documents.append(zip(lemmedTokens, lemmedTokens[1:]))
			if ngrams == 3:
				documents.append(zip(lemmedTokens, lemmedTokens[1:], lemmedTokens[2:]))
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


