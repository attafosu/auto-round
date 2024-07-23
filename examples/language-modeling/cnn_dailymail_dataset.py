import torch
import numpy as np
import logging
import json
import copy
import os
import pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DATASET")

# from item import OutputItem
from datasets import load_from_disk

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

#TODO: Remove this before submissoin
USE_RANDOM=False
import random
if USE_RANDOM:
    random.seed(9973)


MAX_SAMPLES=13368 # maximum samples available in the dataset
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset_CNN(object):
    """ Dataset class for gpt-j """

    def __init__(self, dataset_path=None, model_checkpoint_path="EleutherAI/gpt-j-6B", total_sample_count=MAX_SAMPLES, pad_inputs=False, max_seq_len=1920):
        self.dataset_path = dataset_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,)

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs
        self.max_seq_len = max_seq_len

    def loadDataset(self):
        """ Loads the dataset into memory """

        with open(self.dataset_path, "r") as fid:
            list_data_dict = json.load(fid)
            self.list_data_dict = copy.deepcopy(list_data_dict)

            self.total_sample_count = min(self.total_sample_count, len(list_data_dict))

            if USE_RANDOM: # TODO: Remove this before submission
                list_data_dict = random.choices(list_data_dict, k=self.total_sample_count)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.dataset = []
        self.input_lens = []
        for example in list_data_dict: #[:self.total_sample_count]:
            source = prompt_input.format_map(example)
            input_sample = self.tokenize_function(source)
            input_ids = input_sample.input_ids
            input_len = input_ids.shape[-1]

            if input_len > self.max_seq_len:
                input_ids = input_ids[:, :self.max_seq_len]
                input_len = input_ids.shape[-1]

            attn_mask = torch.ones(input_len).view(1, input_len)
            if self.pad_inputs:
                pad_size = self.max_seq_len - input_len
                pad_tup = (0, pad_size) # right-padding feedback from inc team 
                input_ids = F.pad(input_ids, pad=pad_tup, value=self.tokenizer.pad_token_id)
                attn_mask = F.pad(attn_mask, pad=pad_tup)

            attn_mask[:, -1] = 0
            
            self.dataset.append({'input_ids': input_ids, 'attention_mask': attn_mask})


    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer([examples], truncation=True, max_length=1919, return_tensors="pt", padding=True)
        return example #example["input_ids"]


    def __getitem__(self, index):
        """ Returns sample at 'index' """
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

