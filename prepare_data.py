import sys
import gc

import pandas as pd

def prepare_data():
    test = pd.read_csv('data/test_essays.csv')
    sub = pd.read_csv('data/sample_submission.csv')

    train = pd.read_csv("data/train_v2_drcat_02.csv", sep=',')
    train = train[['text', 'label']]

    train0 = pd.read_csv("data/argugpt.csv", sep=',')
    train0['label'] = 1
    train0 = train0[['text', 'label']]

    train_df = pd.read_parquet("data/train-00000-of-00001-f9daec1515e5c4b9.parquet")
    train1 = pd.DataFrame()
    train1['text'] = train_df['answer']
    train1['label'] = 1

    train_pbw = pd.read_parquet("data/commonlit.parquet")
    train_pbw['text'] = train_pbw['message']
    train_pbw['label'] = 1
    train_pbw = train_pbw[['text', 'label']]
    train_pbw = train_pbw[:20000]

    train_feature = pd.read_csv("data/test.csv", sep=',')
    train_feature['text'] = train_feature['original_text']
    train_feature = train_feature[['text', 'label']]

    train = pd.concat([train, train0, train1, train_pbw, train_feature], axis=0)
    train = train.drop_duplicates(subset=['text'])
    train.reset_index(drop=True, inplace=True)

    counts = train['label'].value_counts()
    num_0 = counts[0]
    num_1 = counts[1]
    print(num_0)
    print(num_1)

    return train, test, sub