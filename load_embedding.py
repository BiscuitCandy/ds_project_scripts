# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Returning embedding of input text """


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torch.utils.data.dataloader import DataLoader
import csv
import numpy as np
import torch
import h5py
import pdb
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    indexed_path: str = field(
        metadata={"help": "indexed h5 file path"}
    )
    inputtext_path: str = field(
        metadata={"help": "The input text file path"}
    )

def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()[0]
    x = {}
    with h5py.File(data_args.indexed_path, 'r') as f:
        with open(data_args.inputtext_path, 'r') as f_in:
            ii = 0
            print("The number of keys in h5: {}".format(len(f)))
            for i, input in enumerate(f_in):
                entity_name = input.strip()

                embedding = f[entity_name]['embedding'][:]
                # if entity_name in x and np.array(x[entity_name]) != np.array(embedding) :
                #     print(x)
                x[entity_name] = embedding

                #print("entity_name = {}".format(entity_name))
                #print("embedding = {}".format(embedding))
                #ii += 1
                #if ii == 7: break
    
    tsv = open("tilde_data.tsv", 'w+')
    meta_tsv = open("meta_tilde_data.tsv", "w+")

    tsv_writer = csv.writer(tsv, delimiter='\t')
    meta_writer = csv.writer(meta_tsv, delimiter = '\t')

    for i in x :
        meta_writer.writerow(i.split())
        tsv_writer.writerow(np.array(x[i]))

    print("complete")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()