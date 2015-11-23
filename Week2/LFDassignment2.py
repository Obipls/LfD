from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import sys
from time import time


# Open the corpus file, loop through it and for every line create a list op tokens, cut first 3 words off. 
# Then, depending on use_sentiment argument, add binary labels of multi labels. Return the two lists
def read_corpus(corpus_file):
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split()

			documents.append(tokens[3:])
			
			# 6-class problem: books, camera, dvd, health, music, software
			labels.append( tokens[0] )
			
	return documents, labels
	
# a dummy function that just returns its input
def identity(x):
	return x

# Execute read_corpus function with the textfile and sentiment(binary classifying)
# Then create for both lists a train and a testdata part consisting of 75% training and 25% test.


X, Y = read_corpus('trainset.txt')#use_sentiment=True
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
	vec = TfidfVectorizer(preprocessor = identity,
						  tokenizer = identity)
else:
	vec = CountVectorizer(preprocessor = identity,
						  tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
						('cls', KNeighborsClassifier(n_neighbors=25, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1))] )

#DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=12, min_samples_split=16, min_samples_leaf=16, min_weight_fraction_leaf=0.0, max_features=20000, random_state=None, max_leaf_nodes=None, class_weight='auto')
#xvalidator = cross_val_score(classifier,X,y=Y,cv=5,scoring='accuracy') 
t0=time()
# Train the classifier with the two lists (75% of total data, consisting of the tags and their labels)
classifier.fit(Xtrain, Ytrain)
t1 = time()
trainTime = t1- t0
# Apply the freshly trained classifier on the unseens testdata and predict it's label
Yguess = classifier.predict(Xtest)
testTime = time() - t1
totalTime = time()-t0

# Print the achieved score in predicting the labels of the unseen goldstandardized testdata.
#print(accuracy_score(Ytest,Yguess))
print("Time to train: {} Time to Test: {} Total: {}".format(trainTime,testTime,totalTime))
print(classification_report(Ytest, Yguess))
#print(classifier.get_params(Xtest))
#print(classifier.predict_proba(Xtest))
#print(sum(xvalidator)/5)




