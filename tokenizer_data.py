import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

def tokenizer_data(train, test):
    LOWERCASE = False
    VOCAB_SIZE = 50275

    # Creating Byte-Pair Encoding tokenizer
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # Adding normalization and pre_tokenizer
    raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    # Adding special tokens and creating trainer instance
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
    # Creating huggingface dataset object
    dataset = Dataset.from_pandas(test[['text']])
    def train_corp_iter(): 
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]
    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenized_texts_test = []

    for text in tqdm(test['text'].tolist()):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []

    for text in tqdm(train['text'].tolist()):
        tokenized_texts_train.append(tokenizer.tokenize(text))

    def dummy(text):
        return text
    #min_df
    min_df=1 if len(test)<=5 else 2 
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
        tokenizer = dummy, min_df=min_df, 
        preprocessor = dummy,
        token_pattern = None, strip_accents='unicode')

    vectorizer.fit(tokenized_texts_test)

    # Getting vocab
    vocab = vectorizer.vocabulary_

    print(vocab)

    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word', min_df=min_df, 
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode'
                                )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()

    return tf_train, tf_test