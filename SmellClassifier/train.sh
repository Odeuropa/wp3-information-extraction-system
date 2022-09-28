#!/bin/bash

todo="train"

if [ "$todo" == "hyperparam" ];
then
  echo "Hyperparameter search."
  python smell_classifier_train.py --hypsearch \
                                 --lang "english" \
                                 --fold 0 \
                                 --model "bert-base-multilingual-uncased"
elif [ "$todo" == "train" ];
then
  echo "Train the model."
  python smell_classifier_train.py --do_train --do_test \
                                 --lang "english" \
                                 --fold 0 \
                                 --learning_rate 2e-5 \
                                 --train_batch_size 4 \
                                 --train_epochs 3 \
                                 --model "bert-base-multilingual-uncased"

elif [ "$todo" == "test" ];
then
  echo "Test the model."
  python smell_classifier_train.py --do_test \
                                 --lang "english" \
                                 --fold 0 \
                                 --model "bert-base-multilingual-uncased-english-0/checkpoint-2055"
fi
