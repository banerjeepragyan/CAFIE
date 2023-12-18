import os

from transformers import GPT2Tokenizer, AdamW, get_scheduler, GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import torch
import pandas as pd
import sys
sys.path.append('../')

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.util import generate_experiment_id, _is_generative
import json
import numpy as np
import random
import os
from stereoset_evaluation import ScoreEvaluator

import os

model_path = 'gpt2-large'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

runner = CrowSPairsRunner(
model=model,
tokenizer=tokenizer,
input_file="/home/pragyanb/biasbench/data/crows/crows_pairs_anonymized.csv",
bias_type='race',
is_generative=True, 
ratio = 0.5,
lamda = 500,
alpha = 0.99,
ct = 'ab', #'ab','i','fill','rb','n'
softmax_temperature = 1,
thres = 0,
scoring_function = 'tanh',
is_self_debias=False,
)
results = runner()
print(results)

