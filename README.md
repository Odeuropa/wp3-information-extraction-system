# Creation of folds for train/dev/test 

This script `create_folds.py` takes into input a folder with Odeuropa INCEpTION annotations exports (webanno format) and returns the data organized in 10 folds in BERT and Machamp (with labels divided in multiple tasks) formats.

To run the script you need to set the following paramethers:

`--folder`: The input folder containing INCEpTION exports

`--output`: Folder where the folds are saved

Other paramethers are:

`--tasktype`: Type of the output. It can be `BERT`(default) or `MULTITASK`

`--tags`: List of the frame elements to keep in the folds separated by `,`


Usage example:
```
python3 create_folds.py --folder en-webanno/ --output OutputDir --tasktype BERT --tags Smell\\_Word,Smell\\_Source,Quality
```



# Convert English plain texts to Bert format for the classification

Run the script `books_converter.py` on the folder containing the documents you want to use to extract frame elements and convert them in a format readable by the classifier.


`--folder`: The input folder containing the books/document (plain txt, no metadata or tags)

`--output`: The output folder for the converted documents

`--label`: A short label used to assign an ID to the documents (so that later they can be matched with the metadata)

`--books` The script allows to merge multible books into a single file, setting the value to 1 create a file for each book


The script creates a `-meta` file outside the output folder to map the document ID with the original books.


Usage example:
```
python3 books_converter.py  --folder books_folder --output output_folder --label abc --books 100
```


# Odeuropa Smell Classifier

The models are available at this link:

| https://drive.google.com/drive/u/1/folders/1IfRbjNc5nAveRaAwraLpYsZJmqS3fsSM |

To install the required packages:

```
pip install -r requirements.txt
```    

Update run_classifier.sh with the right paths of the input and the output folders, the language, and the model path.

Example change for run_classifier.sh:

```
python smell_classifier.py --model_path "bert-base-italian-uncased-odeuropa" --input_path "input_folder" \
                           --output_path "output_folder" --lang "italian"
```

The input files found in 'input_folder' must be in CONLL format with the columns: 'Document', 'Sentence-Token Num', 'Characters', 'Word', 'Label'. No header should be included to the files. The classifier updates the 'Label' column of each file and saves it in 'output_folder'. 

Afterwards, to run the classifier:

```
bash run_classifier.sh
```

# Extract Frame Elements from Bert prediction

The script `extract_annotations.py` takes as input the folder with the predictions from the classifier and return a tsv with the following columns:

Book - Smell_Word - Smell_Source - Quality - Full_Sentence

Usage example:
```
python3 extract_annotations.py  [predictions_folder] > out.tsv
```

