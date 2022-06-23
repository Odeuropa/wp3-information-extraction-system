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