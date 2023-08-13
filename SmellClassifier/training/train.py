# -*- coding: utf-8 -*-
from torch import cuda

import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForTokenClassification, AutoConfig
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from datasets import load_metric, Dataset
import pandas as pd
import numpy as np
import re

import argparse
import csv
import sys
import time
from os import path
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


device = 'cuda' if cuda.is_available() else 'cpu'
print('Device is:', device)

seed = 22
transformers.set_seed(seed)

"""
Models Used are:
model_checkpoint = "dbmdz/bert-base-italian-uncased" -> italian
model_checkpoint = "bert-base-uncased" -> english
model_checkpoint = "camembert-base" -> french, pjox/dalembert -> historical text. {'learning_rate': 2e-05, 'per_device_train_batch_size': 4, 'num_train_epochs': 4}
model_checkpoint = "GroNLP/bert-base-dutch-cased" -> dutch 'learning_rate': 2e-05, 'per_device_train_batch_size': 8, 'num_train_epochs': 8. emanjavacas/GysBERT
model_checkpoint = "deepset/gbert-base -> german, AH: redewiedergabe/bert-base-historical-german-rw-cased, {'learning_rate': 4e-05, 'per_device_train_batch_size': 8, 'num_train_epochs': 6}.
model_checkpoint = "EMBEDDIA/sloberta-> slovene
"""


def sentence_num(row):
    sentenceNum = row['Sentence-Token'].split("-")[0]
    return sentenceNum


def to_label_id(row, id_dict):
    label = row['Tag']
    if label not in id_dict:
        label = 'O'

    labelId = id_dict[label]
    return labelId


def to_clean_label(row):
    #print(row['Document'], row['Sentence-Token'], row['Chars'], row['Word'].encode('utf-8'), row['Tag'])
    clean_tag = row['Tag'].replace("\\", "").replace("\_","_")
    clean_tag = clean_tag.split('|')[0]
    clean_tag = clean_tag.replace("B-I-", "B-")
    return clean_tag


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


"""The script will extract the label list from the data itself. Please be sure your data and labels are clean.
  3 Labels: ['Smell_Word', 'Smell_Source', 'Quality']
  7 Labels: ['Smell_Word', 'Smell_Source', 'Quality', 'Location', 'Odour_Carrier', 'Evoked_Odorant', 'Time']"""


def read_split_fold(split='train', fold="0", lang="english", label_dict=None):
    #change the path template as needed.
    path = '../../../folds/data_{}/folds_{}_{}.tsv'.format(lang, fold, split)
    print('the file path in read_split_fold: ', path)
#     input("Press Enter to continue...")
    try:
        data = pd.read_csv(path, sep='\t', skip_blank_lines=True,
                           encoding='utf-8', engine='python', quoting=csv.QUOTE_NONE,
                           names=['Document', 'Sentence-Token', 'Chars', 'Word', 'Tag','Empty'], header=None)
    except:
        print(f"Cannot read the file {path}")
        if split == "train":
            sys.exit()
        return None, None

    time.sleep(5)
    data.drop('Empty', inplace=True, axis=1)

    #For the reusability purposes, we still extract the label ids from the training data.
    #print('row[Tag]', end=':')
    data['Tag'] = data.apply(lambda row: to_clean_label(row), axis=1)

    print("Number of tags: {}".format(len(data.Tag.unique())))
    frequencies = data.Tag.value_counts()
    print(frequencies)

    if not label_dict:
        labels_to_ids = {k: v for v, k in enumerate(data.Tag.unique())}
    else:
        labels_to_ids = label_dict
    
    ids_to_labels = {v: k for v, k in enumerate(data.Tag.unique())}

    data = data.astype({"Word": str})

    data['Word'] = data.apply(lambda row: replace_punctuation(row), axis=1)
    data['Tag'] = data.apply(lambda row: to_label_id(row, labels_to_ids), axis=1)
    data['Num'] = data.apply(lambda row: sentence_num(row), axis=1)

    # Important point is that we need unique document+Sentence-Token
    data = data.astype({"Num": int})
    data.set_index(['Document', 'Num'])
    df = data.groupby(['Document', 'Num'])['Word'].apply(list)
    df2 = data.groupby(['Document', 'Num'])['Tag'].apply(list)
    mergeddf = pd.merge(df, df2, on=['Document', 'Num'])
    mergeddf.rename(columns={'Word': 'sentence', 'Tag': 'word_labels'}, inplace=True)

    print("Number of unique sentences: {}".format(len(mergeddf)))

    return mergeddf, labels_to_ids, ids_to_labels


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["sentence"], max_length=512, truncation=True, is_split_into_words=True)

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


def cn_hp_space(trial):

    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10, log=True)
    }


def main():
    parser = argparse.ArgumentParser(description='Training with Folds')
    parser.add_argument("--lang", help="Languages: english, german, slovene, dutch, multilingual, french, italian",
                        default="english")
    parser.add_argument("--fold", help="Fold Name", default="0")
    parser.add_argument("--hypsearch", help="Flag for Hyperparameter Search", action='store_true')
    parser.add_argument("--do_train", help="To train the model", action='store_true')
    parser.add_argument("--do_test", help="To test the model", action='store_true')
    parser.add_argument("--learning_rate", type=float, help="Learning Rate for training.", default=2e-5)
    parser.add_argument("--train_batch_size", type=int, help="Training batch size.", default=4)
    parser.add_argument("--train_epochs", type=int, help="Training epochs.", default=3)
    parser.add_argument("--model", action='store', default="bert-base-multilingual-uncased",
                        help="Model Checkpoint to fine tune. If none is given, bert-base-multilingual-uncased will be used.")

    args = parser.parse_args()

    model_checkpoint = args.model
    fold = str(args.fold)
    language = str(args.lang).strip().lower()
    
    print(f'Language: {language}, fold: {fold}, model_checkpoint: {model_checkpoint}')
    
    if 'pjox/dalembert' in model_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    if language not in ['english', 'german', 'italian', 'slovene', 'dutch', 'french']:
        raise Exception(f"Language error: {language} is not among the project languages.")

    if args.do_train and args.hypsearch:
        raise Exception(f"Action error: Cannot do hyperparameter search and train in a single run. Please first run"
                        f"hypsearch and with the parameters obtained as the best, run do_train.")

    config = AutoConfig.from_pretrained(model_checkpoint)
    labels_to_ids = config.label2id
    ids_to_labels = config.id2label

    def model_init():
        m = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config)
        m.to(device)
        return m

    if args.hypsearch or args.do_train:
        trn, labels_to_ids, ids_to_labels = read_split_fold(fold=fold, lang=language)
        train_dataset = Dataset.from_pandas(trn, split="train")
        val, _, _ = read_split_fold(fold=fold, lang=language, split="dev", label_dict=labels_to_ids)
        val_dataset = Dataset.from_pandas(val, split="validation")

        print(labels_to_ids)
        tokenized_train = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
        tokenized_val = val_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
        label_list = list(labels_to_ids.values())
        config.label2id = labels_to_ids
        config.id2label = ids_to_labels
        config.num_labels = len(label_list)

    model_name = model_checkpoint.split("/")[-1]

    if args.hypsearch:
        tr_args = TrainingArguments(
            f"{model_name}-{language}-{fold}-hyp",
            evaluation_strategy="epoch",
            #save_strategy="epoch",
            save_strategy="no",
            per_device_eval_batch_size=8,
            warmup_ratio=0.1,
            seed=22,
            weight_decay=0.01
        )
    elif args.do_train:
        tr_args = TrainingArguments(
            f"{model_name}-{language}-{fold}",
            evaluation_strategy="epoch",
            #save_strategy="epoch",
            save_strategy="no",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=args.train_epochs,
            warmup_ratio=0.1,
            seed=22,
            weight_decay=0.01
        )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [ids_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ids_to_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    if args.do_train or args.hypsearch:
        trainer = Trainer(
            model_init=model_init,
            args=tr_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    elif args.do_test:
        #for testing
        if path.exists(f"{model_checkpoint}/{language}-id2label.json"):
            ids_to_labels = json.load(open(f"{model_checkpoint}/{language}-id2label.json", "r"))
            ids_to_labels = {int(k): v for k, v in ids_to_labels.items()}
            labels_to_ids = {v: int(k) for k, v in ids_to_labels.items()}
            config.label2id = labels_to_ids
            config.id2label = ids_to_labels
            label_list = list(labels_to_ids.values())
            config.num_labels = len(label_list)

        m = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config)
        m.to(device)
        trainer = Trainer(m, data_collator=data_collator, tokenizer=tokenizer)

    if args.hypsearch:
        # hyperparam search with compute_metrics: default maximization is through the sum of all the metrics returned
        best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize", hp_space=cn_hp_space)
        best_params = best_run.hyperparameters
        print(f"Best run is with the hyperparams:{best_params}. You either have to find the right run and checkpoint "
              f"from the models saved or retrain with the correct parameters: referring to "
              f"https://discuss.huggingface.co/t/accessing-model-after-training-with-hyper-parameter-search/20081")

    elif args.do_train:
        trainer.train()

    if args.do_test:
        print("TEST RESULTS")
        test, _, _ = read_split_fold(split="test", label_dict=labels_to_ids, lang=language, fold=fold)
        test_dataset = Dataset.from_pandas(test, split="test")
        tokenized_test = test_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer),
                                          batched=True)
        
        #print('test[:1]:', tokenized_test[:1]['sentence'])
        #print('test[:3][sentence]:', tokenized_test[:3]['sentence'])
        #print('test[:3][Document]:', tokenized_test[:3]['Document'])

        predictions, labels, _ = trainer.predict(tokenized_test)
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [ids_to_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ids_to_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        print("true_predictions:", true_predictions[:3])
        print("true_labels:", true_labels[:3])

        results = metric.compute(predictions=true_predictions, references=true_labels)
        print("\n")
        print(results)
        
        data_and_results = {}
        data_and_results['scores'] = results
        data_and_results['sentences_as_tokens'] = tokenized_test['sentence']
        data_and_results['document'] = tokenized_test['Document']
        data_and_results['predictions'] = true_predictions
        data_and_results['gold_labels'] = true_labels
        
        output_file_name = f"{model_checkpoint}_{language}_fld{fold}_lr{args.learning_rate}_tbs{args.train_batch_size}_te{args.train_epochs}.json"
        output_file_name = 'outputs/'+output_file_name.replace('/','_')
        print('output_file_name:', output_file_name)
        
        with open(output_file_name, 'w') as fw:
            json.dump(data_and_results, fw, indent=4, sort_keys=True, ensure_ascii=True, allow_nan=True, cls=NpEncoder)
        


if __name__ == "__main__":
    main()
