# -*- coding: utf-8 -*-
from torch import cuda
import transformers
import csv
import os
from pathlib import Path
import argparse
import numpy as np
import re

from datasets import Dataset
import time
import pandas as pd
import json
import string

from transformers import DataCollatorForTokenClassification,  AutoTokenizer, PreTrainedTokenizerFast
from transformers import AutoModelForTokenClassification, Trainer


def get_file_list(path):
    """ To get the file list from a given directory path"""
    files = os.listdir(path)  # both folders and files
    file_list = []
    for upload_file in files:
        full_path = os.path.join(path, upload_file)
        if os.path.isdir(full_path):
            continue
        file_list.append(full_path)
    return file_list


def sentence_num(row):
    sentenceNum = row['Sentence-Token'].split("-")[0]
    return sentenceNum


def to_label_id(row, id_dict):
    clean_tag = row['Tag'].replace("\\", "").replace("\_","_")
    clean_tag = clean_tag.split("|")[0]
    if clean_tag not in id_dict:
        clean_tag = 'O'
    labelId = id_dict[clean_tag]
    return labelId


def replace_punctuation(row):
    """Error case in Italian: 'bianco', '-', 'gialliccio' -> 'bianco-gialliccio'
    Bert tokenizer uses also punctuations to separate the tokens along with the whitespaces, although we provide the
    sentences with is_split_into_words=True. Therefore, if there is a punctuation in a single word in a CONLL file
    we cannot 100% guarantee the exact same surface realization (necessary to decide on a single label for a single word) 
    after classification for that specific word:
    e.g., bianco-gialliccio becomes 3 separate CONLL lines: 1) bianco 2) - 3) gialliccio
    Things could have been easier and faster if we were delivering simple sentences as output instead of the exact
    CONLL file structure given as input. """
    word = row['Word'].strip()
    if len(word) > 1:
        word = re.sub(r'[^a-zA-Z0-9]', '', word)
    if word is None or word == "" or word == "nan":
        word = " "
    return word


def read_file(path, label_dict):
    try:
        data = pd.read_csv(path, sep='\t', skip_blank_lines=True,
                           encoding='utf-8', engine='python', quoting=csv.QUOTE_NONE,
                           names=['Document', 'Sentence-Token', 'Chars', 'Word', 'Tag'], header=None)
    except:
        print(f"Cannot read the file {path}")
        return None, None

    time.sleep(5)

    labels_to_ids = label_dict
    data = data.astype({"Word": str})
 
    data['Word'] = data.apply(lambda row: replace_punctuation(row), axis=1)
    data['Tag'] = data.apply(lambda row: to_label_id(row, labels_to_ids), axis=1)
    data['Num'] = data.apply(lambda row: sentence_num(row), axis=1)

    data = data.astype({"Num": int})
    data.set_index(['Document', 'Num'])
    df = data.groupby(['Document', 'Num'])['Word'].apply(list)
    df2 = data.groupby(['Document', 'Num'])['Tag'].apply(list)
    mergeddf = pd.merge(df, df2, on=['Document', 'Num'])
    mergeddf.rename(columns={'Word': 'sentence', 'Tag': 'word_labels'}, inplace=True)

    print("Number of unique sentences: {}".format(len(mergeddf)))

    return data, mergeddf


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["word_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="bert-base-uncased-odeuropa",
        action='store',
        type=str,
        help="A fine-tuned bert-base-uncased checkpoint for labeling smell frame elements."
    )

    parser.add_argument(
        "--input_path",
        default="outfolder_single",
        action='store',
        type=str,
        help="Input folder containing the conll files."
    )

    parser.add_argument(
        "--output_path",
        default="output",
        action='store',
        type=str,
        help="Output folder for the labeled content."
    )
    
    parser.add_argument(
        "--lang",
        default="english",
        action='store',
        type=str,
        help="Model Language: supported languages 'english', 'dutch', 'italian', 'slovene', 'french', 'german'"
    )
    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Device is {device}.")

    seed = 22
    transformers.set_seed(seed)
    model_checkpoint = args.model_path
    folder_path = args.input_path
    output_path = args.output_path
    language = args.lang.lower().strip()

    if language not in ['english', 'german', 'italian', 'slovene', 'dutch', 'french']:
        raise Exception(f"Language error: {language} is not among the project languages.")

    ids_to_labels = json.load(open(f"{model_checkpoint}/{language}-id2label.json", "r"))
    ids_to_labels = {int(k): v for k, v in ids_to_labels.items()}
    labels_to_ids = {v: int(k) for k, v in ids_to_labels.items()}

    Path(output_path).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    label_all_tokens = True
    data_collator = DataCollatorForTokenClassification(tokenizer)

    label_list = list(labels_to_ids.values())

    print("Loading the model checkpoint...")
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model.to(device)
    # Define Trainer for prediction
    test_trainer = Trainer(model, data_collator=data_collator,  tokenizer=tokenizer)

    file_list = get_file_list(folder_path)
    file_list = list(filter(lambda file: os.stat(file).st_size > 0, file_list))

    print("Running the prediction for each file:")
    for f in file_list:
        print("Labeling {}...".format(f), end=' ')
        raw_data, test = read_file(f, label_dict=labels_to_ids)
        if test is None:
            continue
        test_dataset = Dataset.from_pandas(test)
        tokenized_test = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_all_tokens),
                                          batched=True)
        #tokenized_test features: ['sentence', 'word_labels', 'Document', 'Num', 'input_ids', 'token_type_ids',
        # 'attention_mask', 'labels']
        predictions, labels, _ = test_trainer.predict(tokenized_test)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [ids_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_tokens = [[tokenizer.convert_ids_to_tokens(tok) for (tok, l) in zip(elem, label) if l != -100]
                       for elem, label in zip(tokenized_test["input_ids"], labels)]
        preds = []
        toks = []
        for tok, lab, inps in zip(true_tokens, true_predictions, test['word_labels']):
            p = []
            for t, l in zip(tok, lab):
                if language in ["french", "slovene"]: # for roberta based models, the subword signs are different than bert based models
                    if not t.startswith("▁"):
                        toks[-1] = toks[-1] + t
                        continue
                elif t.startswith("##"): # for bert based models
                    toks[-1] = toks[-1] + t.replace("##", "")
                    continue

                p.append(l)
                toks.append(t)

            # if the sentence is problematic and too long, it won't fit to the model so the predictions will be shorter
            # pad it to the length of the actual input
            if len(inps) > len(p):
                p.extend((len(inps)-len(p)) * ['O'])
                
            preds.extend(p)

        # adding the labels to the original data
        raw_data["Tag"] = preds

        # Converting back to the CONLL style
        def blanks(x):
            x.loc[-1] = np.nan
            return x

        raw_data = raw_data.astype(str).groupby(['Document', 'Num'], as_index=True).apply(blanks)
        raw_data.drop('Num', inplace=True, axis=1)

        name = Path(f).name
        output_file = f"{output_path}/{name.split('.')[0]}-output.tsv"
        raw_data.to_csv(output_file, index=False, header=False, sep="\t")
        print("Done. Output file is {}.".format(output_file))


if __name__ == "__main__":
    main()
