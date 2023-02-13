# odeuropa-train-classifier
***To install the requirements:***
pip install -r requirements.txt

***To Train a new model:***
Please check the train.sh file. 

```
bash train.sh
```

There are 3 possible uses of the train.py: 

1. Hyperparameter search - Use --hypsearch option with fold number, language, and the pretrained model to finetune. Get the best run parameters from the output and use them later for the actual training or find the right model among the hyperparameter search model runs. Folder name template is {model_name}-{language}-{fold}-hyp/run-N .
2. Train -  Use --do_train option with fold number, language, learning_rate, train_batch_size, train_epochs and the pretrained model to finetune. To get the results on test set, --do_train can be used together with --do_test. In this case, the script will test the newly fine tuned model on the test of the given fold.
3. Test - Use --do_test option "alone" with language, fold, and ''fine-tuned model''. This will only give the test scores. In order to get the predictions as an output file, please use run_classifier.sh. 

Pretrained model checkpoints for each language:
- "dbmdz/bert-base-italian-uncased" -> italian
- "bert-base-uncased" -> english
- "camembert-base" -> french
- "GroNLP/bert-base-dutch-cased" -> dutch
- "deepset/gbert-base -> german
- "EMBEDDIA/sloberta-> slovene

Data Folders should be in the format below:
    f'data_{language}/folds_{fold}_{split}.tsv'
    
Splits are expected to be train, test, and dev ( e.g., data_english/folds_0_train.tsv or data_french/folds_4_test.tsv).
If you need any other template for the folders, please change the line 82 in train.py depending on your needs.

Suggestion to train models or hyperparam search: First activate your docker environment. Then call:

```
docker exec -it docker-container-name sh
```

This will bring you inside the docker runtime. Then call:

```
nohup bash train.sh &
```

By doing so, you can run the training detached from your session. You can simply exit from the cluster.
When it is finished, you will get all the system out in the nohup file in your working directory.
