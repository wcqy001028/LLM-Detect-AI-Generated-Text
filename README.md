# LLM-Detect-AI-Generated-Text

## ARCHIVE CONTENTS
prepare_data.py           : combine data from different sources <br/>
tokenizer_data.py         : transform the data by TF-IDF<br/>
model_ensemble.py         : create ensemble model<br/>
train_predict.py          : code to rebuild models from scratch and generate predictions


## HARDWARE: (The following specs were used to create the original solution)
Ubuntu 20.04.6 LTS<br/>
CPU RAM 30G

## SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.10.13

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
## below are the shell commands used in each step, as run from the top level directory

```
mkdir -p data/
cd data/
kaggle competitions download -c  llm-detect-ai-generated-text
unzip llm-detect-ai-generated-text.zip
kaggle datasets download -d thedrcat/daigt-v2-train-dataset
unzip daigt-v2-train-dataset.zip
kaggle datasets download -d alejopaullier/argugpt
unzip argugpt.zip
kaggle datasets download -d kagglemini/train-00000-of-00001-f9daec1515e5c4b9
unzip train-00000-of-00001-f9daec1515e5c4b9.zip
kaggle datasets download -d pbwic036/commonlit-data
unzip commonlit-data.zip
kaggle datasets download -d wcqyfly/argu-train
unzip argu-train.zip
cd ..
```

## Train and Predict

**If the number of data  in the test.csv is less than 5, the min_df is set to 1 and the model is not trained which only used for debugging. Conversely, when the number of data in test.csv is greater than 5, the min_df is set to 2 and the model will be trained and will generate prediction results.** <br/>

```
python train_predict.py
```

or just folk the following code and run it to get submission<br/>
**It should be noted that because the number of test sets is less than 3, running all directly will cause the code to report an error, but after submitting, when the test set is replaced with a hidden test set, the code will be run correctly and get the result.** <br/> 
https://www.kaggle.com/code/wcqyfly/fork-of-fork-of-fork-of-llm-daigt-analyse-e-db6333

## The following code is used to combine data from different sources
https://www.kaggle.com/code/wcqyfly/notebook95c85fa3c6
