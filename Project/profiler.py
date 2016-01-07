#!/usr/bin/python3.5

import sys, os, numpy, re
import xml.etree.ElementTree as ET
from lxml import etree
from collections import namedtuple, Counter
from keras.preprocessing.text import *
from basic_neural_network import NNclassify

def main(path):
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
			if filename.endswith('xml'):
				print(filename)
				XMList=[]
				parser=etree.XMLParser(recover=True)
				tree=ET.parse(open(path+'/'+filename,'r', encoding='utf-8', errors="surrogateescape"),parser=parser)
				root=tree.getroot()
				for child in root:
					XMList.append(one_hot(child.text, 100, filters=base_filter(), lower=True, split=" "))
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
		X,Y=main(path)
		if sys.argv[1]=='testing':
			test=True


	else:
		print('Usage: profiler.py <training/testing> <language/all>',langs)



	split=int(0.8*(len(X)))
	Xtrain=X[:split]
	YgenTrain=[Y.Gender for Y in Y][:split]
	YageTrain=[Y.Age for Y in Y][:split]
	Xtest=X[split:]
	YgenTest=[Y.Gender for Y in Y][split:]
	YageTest=[Y.Age for Y in Y][split:]
	print("Test (gen,age):")
	print(Counter(YgenTest))
	print(Counter(YageTest))
	print("Train (gen,age):")
	print(Counter(YgenTrain))
	print(Counter(YageTrain))


	gender=NNclassify(Xtrain,Xtest,YgenTrain,YgenTest,'binary')
	age=NNclassify(Xtrain,Xtest,YageTrain,YageTest,'categorical')

	if test:
		Xrev, Yrev= main('testing/'+sys.argv[2])
		newTruth=open('testing/'+sys.argv[2]+'/truth.txt', 'a')
		for i,rev in enumerate(Yrev):
			newGen=rev._replace(Gender=gender[i])
			newAge=newGen._replace(Age=age[i])
			newTruth.write(newAge)
		newTruth.close()


