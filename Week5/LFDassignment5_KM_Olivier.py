#!/usr/bin/env python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, adjusted_rand_score, v_measure_score, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import sys, numpy
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
def read_corpus(corpus_file):
	wnl = SnowballStemmer('english',ignore_stopwords=True)
	ngrams = 1
	documents=[]
	labels=[]
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens=line.strip().split()
			lemmedTokens = [wnl.stem(token) for token in tokens[3:]]
			if ngrams == 1:
				documents.append(tokens[3:])
			if ngrams == 2:
				documents.append(zip(tokens[3:], tokens[3:][1:]))
			if ngrams == 3:
				documents.append(zip(lemmedTokens, lemmedTokens[1:], lemmedTokens[2:]))
			labels.append(tokens[1])
	return documents, labels

X, Y = read_corpus('musicbooks.txt')#'all_sentiment_shuffled.txt')# )


vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
km = KMeans(n_clusters=2, n_init=10, verbose=1, precompute_distances='auto')
classifier = Pipeline([('vec',vec),('cls', km)])
classifier.fit(X)
Yguess = classifier.predict(X)


labelDict = {}
clusterCombos = defaultdict(list)
for pred, gold in zip(Yguess, Y):
	clusterCombos[pred].append(gold)
for pred, gold in clusterCombos.items():
	labelDict[pred]=Counter(gold).most_common(1)[0][0]
predList = [labelDict[label] for label in Yguess]

print("Rand index: {}".format(adjusted_rand_score(Y,Yguess)))
print("V-measure: {}".format(v_measure_score(Y,Yguess)))
print("All three: {}".format(homogeneity_completeness_v_measure(Y,Yguess)))

cm=confusion_matrix(Y, predList, labels=list(set(Y)))
print(cm)

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of binary label K-Means classification')
plt.colorbar()
tick_marks = numpy.arange(len(list(set(Y))))
plt.xticks(tick_marks, list(set(Y)), rotation=45)
plt.yticks(tick_marks, list(set(Y)))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
