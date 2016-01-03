#!/usr/bin/python3.5

import sys, os, numpy
import xml.etree.ElementTree as ET
from lxml import etree
from collections import namedtuple
from keras.preprocessing.text import *


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
							gold.append((label.Gender,label.Age))
				if XMList not in documents:
					documents.extend(XMList)
	
	return documents,gold


if __name__ == '__main__':
	langs=['dutch','english', 'spanish', 'italian']
	if len(sys.argv) ==3:
		if sys.argv[2]=='all':
			for i,lang in enumerate(langs):
				path=sys.argv[1]+'/'+lang
				X,Y=main(path)

		else:
			path=sys.argv[1]+'/'+sys.argv[2]
			X,Y=main(path)
	else:
		print('Usage: profiler.py <training/testing> <language/all>',langs)

	split=int(0.8*(len(X)))
	Xtrain=X[:split]
	Ytrain=Y[:split]
	Xtest=X[split:]
	Ytest=Y[split:]
	numpy.save('Xtrain',Xtrain)
	numpy.save('Xtest',Xtest)
	numpy.save('Ytrain',Ytrain)
	numpy.save('Ytest',Ytest)
