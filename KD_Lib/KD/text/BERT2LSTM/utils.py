# coding: utf-8

from __future__ import unicode_literals, print_function

from contextlib import closing
from multiprocessing import Pool

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from tqdm import tqdm
import numpy as np
import random

from torchtext import data


class InputExample(object):
    """
    A single training/test example for sequence classification.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def example_to_feature(example_row):
    """
    :param example_row: tuple (example, label_map, tokenizer, max_seq_length)
    :return: InputFeatures
    """

    (example, label_map, tokenizer, max_seq_length) = example_row

    try:
        tokens_a = tokenizer.tokenize(example.text_a)
    except Exception as e:
        tokens_a = []

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[: (max_seq_length - 2)]

    tokens = tokens_a + [tokenizer.sep_token]
    segment_ids = [0] * len(tokens)

    tokens = [tokenizer.cls_token] + tokens
    segment_ids = [0] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_id,
    )


def features_to_dataset(features):
    """
    :param features: list InputFeatures
    :return: TensorDataset
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    return dataset


def df_to_dataset(df, tokenizer, max_seq_length):
    bert_df = pd.DataFrame(
        {
            "id": range(len(df)),
            "label": df.iloc[:, 1],
            "alpha": ["a"] * df.shape[0],
            "text": df.iloc[:, 0].replace(r"\n", " ", regex=True),
        }
    )
    examples = []
    for (i, line) in enumerate(bert_df.T.to_dict().values()):
        guid = "%s-%s" % ("train", i)
        text_a = line["text"]
        label = line["label"]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
        )

    label_map = {label: i for i, label in enumerate([0, 1])}
    examples = [(example, label_map, tokenizer, max_seq_length) for example in examples]

    with closing(Pool(10)) as p:
        features = list(
            tqdm(
                p.imap(example_to_feature, examples, chunksize=100), total=len(examples)
            )
        )
        p.terminate()

    return features_to_dataset(features)


def batch_to_inputs(batch):
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "labels": batch[3],
    }

    return inputs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad(seq, max_len):
    if len(seq) < max_len:
        seq = seq + ["<pad>"] * (max_len - len(seq))

    return seq[0:max_len]


def to_indexes(vocab, words):
    return [vocab.stoi[w] for w in words]


def to_dataset(x, y_real):
    torch_x = torch.tensor(x, dtype=torch.long)
    # torch_y = torch.tensor(y, dtype=torch.float)
    torch_real_y = torch.tensor(y_real, dtype=torch.long)
    return TensorDataset(torch_x, torch_real_y)


def get_essentials(train_df, max_seq_length=128, train_batch_size=16):

    X, y = train_df.iloc[:, 0].values, train_df.iloc[:, 1].values

    text_field = data.Field()
    text_field.build_vocab(X, max_size=10000)

    X_split = [t.split() for t in X]

    # pad
    X_pad = [pad(s, max_seq_length) for s in X_split]

    # to index
    X_index = [to_indexes(text_field.vocab, s) for s in X_pad]

    train_dataset = to_dataset(X_index, y)

    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        drop_last=True,
    )

    return text_field, train_loader
