import argparse
import os
import re

# path  = sys.argv[1]

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Input folder containing texts", metavar="FOLDER", required=True)
parser.add_argument("--output", help="Output folder", metavar="FOLDER", required=True)
parser.add_argument("--books", help="Number of books to aggregate", metavar='N', type=int, default=100, required=False)
parser.add_argument("--label", help="NLabel for the ID", type=str, required=True)

args = parser.parse_args()

path = args.folder
booksnumber = args.books
outPath = args.output
labelID = args.label

isExist = os.path.exists(outPath)
if not isExist:
    os.makedirs(outPath)

metaFileName = outPath
metaFileName = metaFileName.rstrip("/")
metaFileName = metaFileName.strip("\\")
metaFileName = metaFileName + "-meta.tsv"

book_counter = 0
out_name_counter = 0

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith(".txt"):

            with open(os.path.join(root, name), 'r') as f:

                book_counter += 1

                with open(metaFileName, 'a') as metaOut:
                    metaOut.write(labelID + str(book_counter) + "\t" + name + "\n")

                if book_counter % booksnumber == 0:
                    out_name_counter += 1

                out_file_name = "books_merged_" + str(out_name_counter)

                sentence_counter = 1
                word_counter = 1

                outFile = open(os.path.join(outPath, out_file_name) + ".tsv", "a")
                for line in f:
                    line = line.strip("\n")
                    if len(line) < 1:
                        continue
                    accented_chars = 'àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝâêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿÄËÏÖÜŸçÇßØøÅåÆæœ'
                    line = re.sub(r'\t', ' ', line)
                    line = re.sub(r'([^a-zA-Z' + accented_chars + '0-9])', ' \\1 ', line)
                    line = re.sub(' +', ' ', line)
                    parts = line.split(" ")

                    for token in parts:
                        if len(token)<1:
                            continue
                        tokenID = str(sentence_counter) + "-" + str(word_counter)
                        outFile.write(labelID + str(book_counter) + "\t" + tokenID + "\t-\t" + token + "\tO\n")

                        word_counter += 1
                        if token == ".":
                            word_counter = 1
                            sentence_counter += 1
                            outFile.write("\n")
                f.close()
