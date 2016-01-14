#!/usr/bin/python3.5

import sys, os, numpy, re
import xml.etree.ElementTree as ET
from lxml import etree
from collections import namedtuple, Counter
from keras.preprocessing.text import *
from basic_neural_network import NNclassify
from sklearn.cross_validation import train_test_split as TTS
from nltk.stem.snowball import SnowballStemmer
from itertools import chain


def main(path,language):
	SS = SnowballStemmer(language)
	documents=[]
	labels=[]
	gold=[]

	with open(path+'/'+'truth.txt', encoding='utf-8') as f:
		for line in f:
			properties=line.strip().split(':::')
			properties=[0 if x=='F' else 1 if x=='M' else x for x in properties]
			properties=[0 if x=='XX-XX' else 1 if x=='18-24' else 2 if x=='25-34' else 3 if x=='35-49' else 4 if x=='50-XX' else x for x in properties]
			User=namedtuple('User', 'ID, Gender, Age, Ratio1, Ratio2, Ratio3, Ratio4, Ratio5')
			user=User(*properties)
			labels.append(user)


	#Loop over all XML's
	for subdir, dirs, files in os.walk(path):
		for filename in files:
			ngramList=[]
			if filename.endswith('xml'):
				#print(filename)
				XMList=[]
				parser=etree.XMLParser(recover=True)
				tree=ET.parse(open(path+'/'+filename,'r', encoding='utf-8', errors="surrogateescape"),parser=parser)
				root=tree.getroot()
				for child in root:
					lemmedTokens = [SS.stem(token) for token in child.text.split()]
					bigrams = list(chain.from_iterable(zip(lemmedTokens,lemmedTokens[1:])))
					XMList.append(one_hot(" ".join(bigrams), 1000, filters=base_filter(), lower=True, split=" "))
					for label in labels:
						if label.ID == filename[:-4]:
							gold.append(label)
				if XMList not in documents:
					documents.extend(XMList)
	
	return documents,gold


if __name__ == '__main__':
	test = False
	langs=['dutch','english', 'spanish', 'italian']
	if len(sys.argv) ==3:
		path='training/'+sys.argv[2]
		X,Y=main(path,sys.argv[2])
		if sys.argv[1]=='testing':
			test=True
	else:
		print('Usage: profiler.py <training/testing> <language/all>',langs)



	if test:
		Xrev, Yrev= main('testing/'+sys.argv[2],sys.argv[2])
		newTruth=open('testing/'+sys.argv[2]+'/truth.txt', 'a')
		for i,rev in enumerate(Yrev):
			newGen=rev._replace(Gender=gender[i])
			newAge=newGen._replace(Age=age[i])
			newTruth.write(newAge)
		newTruth.close()

	else:
		X_train, X_test, y_train, y_test = TTS(X, Y, test_size=0.2, random_state=0)
		YgenTrain=[Y.Gender for Y in y_train]
		YageTrain=[Y.Age for Y in y_train]
		YgenTest=[Y.Gender for Y in y_test]
		YageTest=[Y.Age for Y in y_test]
		print("Test (gen,age):")
		print(Counter(YgenTest))
		print(Counter(YageTest))
		print("Train (gen,age):")
		print(Counter(YgenTrain))
		print(Counter(YageTrain))


		gender=NNclassify(X_train,X_test,YgenTrain,YgenTest,'binary')
		age=NNclassify(X_train,X_test,YageTrain,YageTest,'categorical')


