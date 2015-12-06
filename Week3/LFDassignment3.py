from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy, pickle, os
from collections import Counter
import matplotlib.pyplot as plt

# Read in the NER data, with either 2 or 6 classes
def read_corpus(corpus_file, binary_classes):
    print('Reading in data...')
    words = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()

            words.append(parts[0])
            if binary_classes:
                if parts[1] == 'GPE' or parts[1] == 'LOC':
                    labels.append('LOCATION')
                else:
                    labels.append('NON-LOCATION')
            else:
                labels.append(parts[1])
                
    print('Done!')
    return words, labels

# Read in word embeddings 
def read_embeddings(embeddings_file):
    embeddings = {}
    print('Reading in embeddings...')
    if 'embeddings.pickle' in os.listdir():
        embeddings = pickle.load(open('embeddings.pickle', 'rb'))
        print('Done!')
        return embeddings
    with open(embeddings_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            embeddings[line.split(' ')[0]] = numpy.array(line.split(' ')[1:], dtype=numpy.float64)
    # Calculate an average vector to use for unknown words
    embeddings['UNK'] = numpy.array(list(embeddings.values())).mean(0)
    print('Done!')
    pickle.dump(embeddings, open('embeddings.pickle', 'wb'), -1)
    return embeddings

# Turn words into embeddings
def vectorizer(words, embeddings):
    vectorized_words = []
    for word in words:
        try:
            vectorized_words.append(embeddings[word.lower()])
        except KeyError:
            vectorized_words.append(embeddings['UNK'])
    return numpy.array(vectorized_words)
   

# Read in the data, split into train and test data, and read in the embeddings
X, Y = read_corpus('NER_data.txt', False)
embeddings = read_embeddings('glove.6B.50d.txt')
X = vectorizer(X, embeddings)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# Combine the vectorizer with a Perceptron classifier
classifier = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=20, shuffle=True, verbose=1, eta0=0.1, n_jobs=1, random_state=0, class_weight=None, warm_start=False)

# Train the classifier
classifier.fit(Xtrain, Ytrain)
# Classify the test data  
Yguess = classifier.predict(Xtest)
print('Classification accuracy: %s' % (accuracy_score(Ytest, Yguess)))
cm=confusion_matrix(Ytest, Yguess, labels=list(set(Y)))
print(cm)
print("Most Frequent Class Baseline= {}, very significant!".format(Counter(Ytrain).most_common(1)[0][1]/sum(Counter(Ytrain).values())))
print(classifier.predict(vectorizer('Air Force',embeddings)))

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix of multi label Perceptron classification')
plt.colorbar()
tick_marks = numpy.arange(len(list(set(Y))))
plt.xticks(tick_marks, list(set(Y)), rotation=45)
plt.yticks(tick_marks, list(set(Y)))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()