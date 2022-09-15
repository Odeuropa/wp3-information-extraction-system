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

Update run_classifier.sh with the right paths and run it to label all the files that can be found under input_path.

Example change for run_classifier.sh:

```
python smell_classifier.py --model_path "bert-base-italian-uncased-odeuropa" --input_path "input_folder" \
                           --output_path "output_folder" --lang "italian"
```

# Extract Frame Elements from Bert prediction

The script `extract_annotations.py` takes as input the folder with the predictions from the classifier and return a tsv with the frame elements, the sentences from which they are extracted and the associated books.

`--folder`: The input folder containing the predictions of the classifier

`--output`: File to save the annotations

`--stopwords`: file containing stopwords to be ignored during the extraction (optional)

`--smellwordtag`: Label used for the smell word (usualle `Smell_Word` or `Smell\_Word`)

`--tags`: List of the frame elements to extract separated by `,`

Usage example:
```
python3 extract_annotations.py --folder [predictions_folder] --smellwordtag Smell_Word --tags Smell_Source,Quality --stopwords stopwords.txt --output out.tsv
```

