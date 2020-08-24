import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torch.utils.data import TensorDataset

"""
DATALOADER UTILITIES
"""


def get_bert_dataloader(df, tokenizer, max_seq_length=64, batch_size=16, mode="train"):
    """
    Helper function for generating dataloaders for BERT
    """

    dataset = df_to_bert_dataset(df, max_seq_length, tokenizer)

    if mode == "validate":
        val_sampler = SequentialSampler(dataset)
        val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size)
        return val_loader

    if mode == "distill":
        distill_sampler = SequentialSampler(dataset)
        distill_loader = DataLoader(
            dataset, sampler=distill_sampler, batch_size=batch_size
        )
        return distill_loader

    elif mode == "train":
        train_sampler = RandomSampler(dataset)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        return train_loader


def df_to_bert_format(df, max_length, tokenizer):
    sentences = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(encoded_dict["input_ids"])
        attention_masks.append(encoded_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def df_to_bert_dataset(df, max_length, tokenizer):
    input_ids, attention_masks, labels = df_to_bert_format(df, max_length, tokenizer)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset
