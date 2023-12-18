import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
sys.path.append('../')
import argparse
import json
import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from biasbench.experiments.scoring import ScoringAlgo

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
parser = argparse.ArgumentParser(description="Blah")

parser.add_argument(
    "--alp",
    default=0.99
)
parser.add_argument(
    "--lmd",
    default=100
)
parser.add_argument(
    "--model_name",
)


# device = "cuda:7" if torch.cuda.is_available() else "cpu"
args = parser.parse_args()

sa = SentimentIntensityAnalyzer()

# model_name = 'EleutherAI/pythia-6.9b'
# print(model_name)

model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

male_words = []
female_words = []
fw2 = []
male_word_path = ""
with open(male_word_path, "r") as f:
    for line in f:
        male_words.append(line[:-1])
female_word_path = ""
with open(female_word_path, "r") as f:
    for line in f:
        female_words.append(line[:-1])
female_word_path = ""
with open(female_word_path, "r") as f:
    for line in f:
        fw2.append(line[:-1])

def generator(prompt, max_new_tokens=10):
    # print(prompt)
    # import ipdb; ipdb.set_trace()
    try:
        runner = ScoringAlgo(
            mdl=model,
            model_name_path='g',
            tokenizer=tokenizer,
            _do_sdb=False, #add sdb prefix to the sentence
            ratio=0.5, #0.5 for avg
            scoring_function="tanh", #Other options- avg, jpdf, arctan, weight
            threshold=0, #bias threshold for scoring
            lmbda=int(args.lmd), #when using scoring, hyperparamenter for scoring function (increasing it increses debiasing but reduces LM score)
            alpha_ratio=float(args.alp), #new probs = alpha*debiased_probs + (1-alpha)*vanilla_probs
            softmax_temperature=1, #temperature for vanilla_probs in alpha
            prompt=prompt[0],
            context="",
            act1=male_words,
            cnt1=female_words,
            ct2=fw2,
            sent_len=max_new_tokens,
            batch_size=1,
            max_seq_length=128,
            bias_type=None,
            words_to_ignore = []
        )
        output, sc, gen_sent, _ = runner()
    except:
        gen_sent = prompt
        print(prompt)
    return gen_sent

prompt = "Two boys start a"
print(generator(prompt))

