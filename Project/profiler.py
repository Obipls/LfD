#!/usr/bin/python3.5

import sys, os
import xml.etree.ElementTree as ET
from lxml import etree


def main(usage,language):
	#Loop over all XML's
	for subdir, dirs, files in os.walk(usage):
		for filename in files:
			if filename.endswith('xml'):
				print(filename,subdir[9:])
				parser = etree.XMLParser(recover=True)
				tree=ET.parse(open(usage+'/'+subdir[9:]+'/'+filename,'r', encoding='utf-8', errors="surrogateescape"),parser=parser)
				root=tree.getroot()
				if root != None:
					for child in root:
						print(child.text)





if __name__ == '__main__':
	if len(sys.argv) ==3:
		usage=sys.argv[1]
		language=sys.argv[2]
		main(usage,language)
	else:
		print("Usage: profiler.py <training/testing> <language>")