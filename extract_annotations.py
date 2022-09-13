import sys
import os

# inputFile = sys.argv[1]
path = sys.argv[1]

tokenID =1
textIndex = 3




smell_word_tag = "Smell\\_Word"
frameElements = ["Smell\\_Source","Quality"]

def pocess_annotations(myAnnotations:list):
	annotationsDict = dict()
	outputDict = dict()

	s = [a for a in myAnnotations if smell_word_tag in " ".join(a)]

	# save all annotations in annotationsDict
	if len(s) == 0:
		return(None)
	annotationsDict[smell_word_tag] = s
	
	for f in frameElements:
		fe = [a for a in myAnnotations if f in " ".join(a)]
		annotationsDict[f] = fe

	#parse the annotations in annotationsDict merging tokens in the same span and puth them in outputDict
	
	# smell words:

	outputDict[smell_word_tag] = []
	firstID = int(annotationsDict[smell_word_tag][0][tokenID].split("-")[1])-1

	span = []
	for sw in annotationsDict[smell_word_tag]:

		myID = sw[tokenID].split("-")[1]
		myToken = sw[textIndex]

		if int(myID) == int(firstID)+1:
			span.append(myToken)
			firstID = myID

		else:	
			outputDict[smell_word_tag].append(" ".join(span))
			firstID = myID
			span = []
			span.append(myToken)

	outputDict[smell_word_tag].append(" ".join(span))

	
	#frame-elements

	

	for frameElement in frameElements:
		span = []
		outputDict[frameElement] = []
		if len(annotationsDict[frameElement]) == 0:
			continue
		
		firstID = int(annotationsDict[frameElement][0][tokenID].split("-")[1])-1
		
		for f in annotationsDict[frameElement]:

			myID = f[tokenID].split("-")[1]
			myToken = f[textIndex]
			
			if int(myID) == int(firstID)+1:
				span.append(myToken)
				firstID = myID
		
			else:	
				if " ".join(span).lower() not in spamList:
					outputDict[frameElement].append(" ".join(span))
				firstID = myID
				span = []
				span.append(myToken)

		if " ".join(span).lower() not in spamList:
			outputDict[frameElement].append(" ".join(span))
	
	return(outputDict)

def dictToString (myDict):
	myList = []
	myList.append("|".join(myDict[smell_word_tag]))
	for f in frameElements:
		myList.append("|".join(myDict[f]))
	return("\t".join(myList)+"\t")



spamList = []
with open("stopwords.txt", 'r') as file:
	for line in file:
		line = line.strip("\n")
		spamList.append(line)

annotations_list = []
sentence_list = []


print("Book\tSmell_Word\tSmell_Source\tQuality\tFull_Sentence")


for root, dirs, files in os.walk(path):
	for name in files:
		if name.startswith("."):
			continue
		with open(os.path.join(root,name), 'r') as file:

			for line in file:
				line = line.strip()
				parts = line.split("\t")
				
				
				if line == "":
					if len(annotations_list) > 1:
						
						dictAnnotations = pocess_annotations(annotations_list)
						
						if dictAnnotations != None:
							stringToPrint = dictToString(dictAnnotations)
							print(title+"\t"+stringToPrint+"\t"+" ".join(sentence_list))
							if "\t\t" not in stringToPrint:
								print(title+"\t"+stringToPrint+"\t"+" ".join(sentence_list))
					sentence_list = []
					annotations_list = []
					continue

				title = parts[0]

				sentence_list.append(parts[textIndex])
				for p in parts[textIndex+1:]:
					if p != "O":
						annotations_list.append(parts)
						continue