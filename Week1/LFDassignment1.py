from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import sys


# Open the corpus file, loop through it and for every line create a list op tokens, cut first 3 words off. 
# Then, depending on use_sentiment argument, add binary labels of multi labels. Return the two lists
def read_corpus(corpus_file, argv):
	pos=neg=books=camera=health=dvd=music=software=0
	documents = []
	labels = []
	with open(corpus_file, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split()

			documents.append(tokens[3:])

			if argv[1]=="binary":
				if tokens[1] == "pos":
					pos+=1
				else:
					neg+=1

				# 2-class problem: positive vs negative
				labels.append( tokens[1] )
			elif argv[1]=="multi":
				if tokens[0] == "books":
					books+=1
				elif tokens[0] == "camera":
					camera+=1
				elif tokens[0] == "health":
					health+=1
				elif tokens[0] == "music":
					music+=1
				elif tokens[0] == "dvd":
					dvd+=1
				else:
					software+=1

				# 6-class problem: books, camera, dvd, health, music, software
				labels.append( tokens[0] )

	if argv[1]=="binary":
		print("prior pos= ", pos/(pos+neg))
		print("prior neg= ", neg/(pos+neg))
	
	else:
		total=books+camera+health+music+software
		print (books)
		print("prior books= ", books/total)
		print("prior camera= ", camera/total) 	
		print("prior health= ", health/total) 	
		print("prior music= ", music/total) 	
		print("prior software= ", software/total) 	
		print("prior dvd= ", dvd/total) 			
	return documents, labels
	
# a dummy function that just returns its input
def identity(x):
	return x

# Execute read_corpus function with the textfile and sentiment(binary classifying)
# Then create for both lists a train and a testdata part consisting of 75% training and 25% test.
if len(sys.argv) < 2:
	print("Usage: python LfDassignment1.py [binary] or [multi]")
else: 
	X, Y = read_corpus('all_sentiment_shuffled.txt', sys.argv)#use_sentiment=True
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
							('cls', MultinomialNB())] )


	# Train the classifier with the two lists (75% of total data, consisting of the tags and their labels)
	classifier.fit(Xtrain, Ytrain)

	# Apply the freshly trained classifier on the unseens testdata and predict it's label
	Yguess = classifier.predict(Xtest)


	# Print the achieved score in predicting the labels of the unseen goldstandardized testdata.
	print(classification_report(Ytest, Yguess))
	print(classifier.get_params(Xtest))
	print(classifier.predict_proba(Xtest))

	


