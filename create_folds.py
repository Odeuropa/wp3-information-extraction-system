import os
import sys
import re
import  math
from sys import getsizeof


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Input folder containing INCEPTION exports", metavar="FOLDER", required=True)
parser.add_argument("--output", help="Output folders", metavar="FOLDER", required=True)
parser.add_argument("--tasktype", choices={"BERT", "MULTITASK"} , help="BERT or MULTITASK", metavar="TASK", default="BERT", type=str)
parser.add_argument("--tags", help="List of labels comma separated", metavar="TASK", default="Smell\\_Word,Smell\\_Source,Quality", type=str)
args = parser.parse_args()

path = args.folder
tasktype = args.tasktype
foldsNumber  = 10


tagsColumns = []
if tasktype == "BERT":
	myList = []
	for l in args.tags.split(","):
		myList.append(l)
	tagsColumns.append(myList)

if tasktype == "MULTITASK":
	for l in args.tags.split(","):
		myList = []
		myList.append(l)
		tagsColumns.append(myList)


foldsDict = dict()
for n in range(foldsNumber):
	foldsDict[n] = []


## creation dictionary of frame elements
tagsDict = dict()
for root, dirs, files in os.walk(path):
	for name in files:
		if name.endswith(".tsv"):
			
			with open(os.path.join(root,name), 'r') as f:
				for line in f:
					line = line.strip("\n")
					parts = line.split("\t")
					if len(parts)<5:
						continue

					thisTag = parts[3].split("|")        			
					for t in thisTag:
						t = re.sub('\[[0-9]+\]', '', t)
						tagsDict[t] = True

tList = []
for t in tagsDict.keys():
	if len(t) <2:
		continue
	tList.append(t)
tList.sort()

tagIndex = dict()

for n in range(0,len(tList)):
	tagIndex[tList[n]] = n




def addBIOtoBuffer (bufferArray, tlist):
	for n in range(0,len(tList)):
		
		if bufferArray[0][n] == "_" and bufferArray[1][n] == "_":
			continue

		if bufferArray[0][n] != "B-"+bufferArray[1][n] and  bufferArray[0][n] != "I-"+bufferArray[1][n] and bufferArray[0][n] != "_" and  bufferArray[1][n] != "_" :
				bufferArray[0][n] = "B-" + bufferArray[0][n]
				bufferArray[1][n] = "I-" + bufferArray[1][n]
				# print("aaa")
				continue

		if bufferArray[0][n] == "_" and bufferArray[1][n] != "_":
			bufferArray[1][n] = "B-" + bufferArray[1][n]
		if bufferArray[0][n] != "_" and bufferArray[1][n] != "_":
			if bufferArray[0][n] == "B-"+bufferArray[1][n]:
				bufferArray[1][n] = "I-" + bufferArray[1][n]
			if bufferArray[0][n] == "I-"+bufferArray[1][n]:
				bufferArray[1][n] = "I-" + bufferArray[1][n]

	
	for x in range(0,len(bufferArray[0])):
		if bufferArray[0][x].startswith("B-") or  bufferArray[0][x].startswith("I-") or bufferArray[0][x] == "_":
			continue
		else:
			bufferArray[0][x] = "B-"+bufferArray[0][x]
	
	bufferToPrint = re.sub('\[[0-9]+\]', '', "\t".join(bufferArray[0]))
	printableSting = ("\t".join(lineArray[0])+"\t"+bufferToPrint) 
	return(printableSting)




for root, dirs, files in os.walk(path):
	for name in files:
		if name.endswith(".tsv"):
			
			with open(os.path.join(root,name), 'r') as f:

				smellTotal = 0
				for line in f:
					if "Smell\_Word" in line:
						smellTotal +=1
				foldSize = round(smellTotal/foldsNumber)

			buffer = []
			thisDocLinesList = []
			smellCount = 0

			foldIndex = 0
				
			with open(os.path.join(root,name), 'r') as f:
				fileNameToPrint = os.path.join(root,name).replace(path,"")
				fileNameToPrint = re.sub(' .+/', '/', fileNameToPrint)
				for line in f:
					line = line.strip("\n")
					if line == "":
						thisDocLinesList.append(line)
						continue

					parts = line.split("\t")
					
					if len(parts)<5:
						continue

					thisTag = parts[3].split("|")      
					
					newLine = "\t".join(parts[0:3])
					
					for t1 in tList:
						for t2 in thisTag:
							if t1 in t2:
								newLine= newLine+"\t"+t2
							else:
								newLine= newLine+"\t_"
					thisDocLinesList.append(newLine)

				bufferArray = []
				lineArray = []

				
	

				bufferToPrintList = []

				for l in thisDocLinesList:
					
					if l == "":
						if foldSize == 0:
							foldIndex = 0
						else:
							foldIndex = math.floor(smellCount/foldSize)

						if foldIndex > foldsNumber-1: foldIndex = foldsNumber-1
						for item in bufferToPrintList:
							foldsDict[foldIndex].append(item) 
	
						foldsDict[foldIndex].append(l)
						bufferToPrintList = []
						bufferArray = []
						lineArray = []
						continue


					parts = l.split("\t")[3:]
					lineBegin = l.split("\t")[:3]
					lineBegin[0] = fileNameToPrint+"\t"+lineBegin[0]
					
					bufferArray.append(parts)
					lineArray.append(lineBegin)
					
					if len(bufferArray) < 2:
						continue

					bioString = addBIOtoBuffer(bufferArray,tList)
					bufferToPrintList.append(bioString)


					if "Smell\_Word" in bioString: smellCount +=1
					
					del(bufferArray[0])
					del(lineArray[0])
					
					


				#check if there is an annotation on the last word/line of the document:
				try:
					
					for x in range(0,len(bufferArray[0])):
						if bufferArray[0][x].startswith("B-") or  bufferArray[0][x].startswith("I-") or bufferArray[0][x] == "_":
							continue
						else:
							bufferArray[0][x] = "B-"+bufferArray[0][x]
				
					bufferToPrint = re.sub('\[[0-9]+\]', '', "\t".join(bufferArray[0]))

					if "Smell\_Word" in bufferToPrint:
							smellCount +=1

					try:
						foldIndex = math.floor(smellCount/foldSize)
					except:
						foldIndex
					if foldIndex > foldsNumber-1: foldIndex = foldsNumber-1				
					foldsDict[foldIndex].append("\t".join(lineArray[0])+"\t"+bufferToPrint) #####

				except:
					emptyLine = True



def writeFold  (foldID,fileName):
	colums = 4 #number of columns until the word (included)
	array_to_print = []

	for l in foldsDict[foldID]:
		stringToPrint = ""

		if len(l) < 1:
			array_to_print.append("\n")		
			continue
		
		parts = l.split("\t")		

		stringToPrint = stringToPrint + "\t".join(parts[:colums]) + "\t"
		
		for myTags in tagsColumns:

			if len(parts) <2:
				array_to_print.append("\n")
				continue
			tagsToAdd = []
			for t in myTags:
				tagsToAdd.append(parts[colums+tagIndex[t]])
			tagsToAdd2 = []
			if "B-" in " ".join(tagsToAdd) or "I-" in " ".join(tagsToAdd):
				for t in tagsToAdd:
					if t != "_":
						tagsToAdd2.append(t)
			else:
				tagsToAdd2.append("O")
			stringToPrint = stringToPrint + "|".join(tagsToAdd2)
			stringToPrint = stringToPrint + "\t"
		
		array_to_print.append(stringToPrint)
			

	while array_to_print[0] == "" or  array_to_print[0] == "\n" : 
		del(array_to_print[0])


	index = array_to_print[0].split("\t")[1].split("-")[0]


	with open(fileName, 'a') as f:
		for l in array_to_print:
			if l=="\n":
				continue
			if l.split("\t")[1].split("-")[0] != index:
				index = l.split("\t")[1].split("-")[0]
				f.write("\n")	
			f.write(l.replace("\\", ""))
			f.write("\n")
	f.close()


#############
# print files
#############

outputPath = args.output


folderExist = os.path.exists(outputPath)

if not folderExist:
  os.mkdir(outputPath)


for i in range(foldsNumber):
	f = open(os.path.join(outputPath,"folds_"+str(i)+"_train.tsv"), 'w')
	f.close()
	f = open(os.path.join(outputPath,"folds_"+str(i)+"_dev.tsv"), 'w')
	f.close()
	f = open(os.path.join(outputPath,"folds_"+str(i)+"_test.tsv"), 'w')
	f.close()


# testdev = [[9,0],[8,9],[7,8],[6,7],[5,6],[4,5],[3,4],[2,3],[1,2],[0,1]]
testdev = [[9,0],[7,8],[5,6],[3,4],[1,2]]

counter = -1
for pair in testdev:
	counter += 1
	train = []
	for n in range(foldsNumber):
		if n not in pair:
			train.append(n)
	dev = []
	test = []
	dev.append(pair[0])
	test.append(pair[1])

	for fold in train: writeFold(fold,os.path.join(outputPath,"folds_"+str(counter)+"_train.tsv"))
	for fold in dev: writeFold(fold,os.path.join(outputPath,"folds_"+str(counter)+"_dev.tsv"))
	for fold in test: writeFold(fold,os.path.join(outputPath,"folds_"+str(counter)+"_test.tsv"))
