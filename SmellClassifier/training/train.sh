#!/bin/bash

todo="hyperparam"

if [ "$todo" == "hyperparam" ];
then
  echo "Hyperparameter search."
  python train.py --hypsearch \
                                 --lang "french" \
                                 --fold 3 \
                                 --model "camembert-base"
elif [ "$todo" == "train" ];
then
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. english. bert-base-uncased"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "english" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "bert-base-uncased"
#   done
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. english. emanjavacas/MacBERTh"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "english" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "emanjavacas/MacBERTh"
#   done
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. dutch, GroNLP/bert-base-dutch-cased"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "dutch" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "GroNLP/bert-base-dutch-cased"
#   done
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. dutch, emanjavacas/GysBERT"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "dutch" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "emanjavacas/GysBERT"
#   done
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. german, deepset/gbert-base"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "german" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "deepset/gbert-base"
#   done
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. german, redewiedergabe/bert-base-historical-german-rw-cased"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "german" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "redewiedergabe/bert-base-historical-german-rw-cased"
#   done
  
  #### fold 0 'learning_rate': 2e-05, 'per_device_train_batch_size': 4, 'num_train_epochs': 7
  #### fold 1 hyperparams:{'learning_rate': 2e-05, 'per_device_train_batch_size': 4, 'num_train_epochs': 10}
  #### fold 2 {'learning_rate': 5e-05, 'per_device_train_batch_size': 8, 'num_train_epochs': 5}
  for i in 0 1 2 3 4 5 6 7 8 9
  do
   echo "Welcome $i times. french, camembert-base, case -> (RoBERTa based, cased)"
   echo "Train the model."
   python train.py --do_train --do_test \
                                 --lang "french" \
                                 --fold $i \
                                 --learning_rate 5e-05 \
                                 --train_batch_size 8 \
                                 --train_epochs 5 \
                                 --model "camembert-base"
  done
  
#   for i in 0 1 2 3 4 5 6 7 8 9
#   do
#    echo "Welcome $i times. french, cased -> (RoBERTa based, cased)"
#    echo "Train the model."
#    python train.py --do_train --do_test \
#                                  --lang "french" \
#                                  --fold $i \
#                                  --learning_rate 4e-5 \
#                                  --train_batch_size 8 \
#                                  --train_epochs 15 \
#                                  --model "pjox/dalembert"
#   done
  

elif [ "$todo" == "test" ];
then
  echo "Test the model."
  python train.py --do_test \
                                 --lang "english" \
                                 --fold 0 \
                                 --model "bert-base-uncased-english-0/checkpoint-344" \
                                 --learning_rate 4e-5 \
                                 --train_batch_size 8 \
                                 --train_epochs 10

fi                            
