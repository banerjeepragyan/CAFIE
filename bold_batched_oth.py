# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
sys.path.append('../')
import argparse
# %%
import json
import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from biasbench.experiments.scoring import ScoringAlgo
# from biasbench.bias_bench.debias.self_debias.modeling import AutoWrapper
from biasbench.bias_bench.model import models

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
parser = argparse.ArgumentParser(description="Blah")

parser.add_argument(
    "--model_name",
    default="EleutherAI/pythia-6.9b"
)
parser.add_argument(
    "--method",
    default="sd"
)
parser.add_argument(
    "--bias_type",
    default="gender"
)
parser.add_argument(
    "--bias_direction",
    action="store",
    type=str,
    default="/home/pragyanb/biasbench/results/subspace/subspace_m-AutoModelForCausalLM_c-openlm-research/open_llama_13b_t-gender.pt",
    help="Path to the file containing the pre-computed bias direction for SentenceDebias.",
)
# %%
# device = "cuda:7" if torch.cuda.is_available() else "cpu"
args = parser.parse_args()
prefix = "Generate some text that prefers the minority over the majority for traditionally majority associated adjectives and verbs: "
cot_prefix = "Generate some text that treats all groups of people the same and does not discriminate between them. Lets think step by step: "

# %%
sa = SentimentIntensityAnalyzer()
device = "cuda"

DEBIASING_PREFIXES = [
    "The following text discriminates against people because of their race/color: ",
    "The following text discriminates against people because of their religion: ",
    "The following text discriminates against people because of their gender: ",
    "The following text discriminates against people because of their profession: "
]

kwargs = {}
if args.bias_direction is not None:
    # Load the pre-computed bias direction for SentenceDebias.
    bias_direction = torch.load(args.bias_direction)
    kwargs["bias_direction"] = bias_direction

if args.method == 'sdb':
    wrapper = AutoWrapper(model_name=args.model_name)
elif args.method == 'imp' or args.method == 'cot':
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
else:
    model = getattr(models, "SentenceDebiasAutoModelForCausalLM")(
        args.model_name, **kwargs
    ).to(device)

def gen_sdb (text, num_max = 10):
    try:
        sss = wrapper.generate_self_debiasing(
            [text[0]], debiasing_prefixes=DEBIASING_PREFIXES, max_length=num_max
        )
        # print(text[0]+sss[0])
        return text[0]+sss[0]
    except:
        return text[0]

def gen_sd(text, num_max = 10):
    try:
        input_ids = tokenizer(text[0], return_tensors='pt')['input_ids'].to(device)
        sent = text
        for i in range(num_max):
            outputs = torch.nn.functional.softmax(model(input_ids)[0], dim=-1)
            logit = torch.argmax(outputs[-1,-1,:])
            word = tokenizer.decode(logit)
            sent += word
            input_ids = torch.cat([input_ids, logit.unsqueeze(-1).unsqueeze(-1)], dim=-1)
        # print(sent)
        return sent
    except:
        return text[0]

# model_name = 'EleutherAI/pythia-6.9b'
# print(model_name)

# %%
# model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

# %%
# model.to(device)

# %%
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

male_words = []
female_words = []
fw2 = []
male_word_path = "/home/pragyanb/biasbench/experiments/act.txt"#"/home/pragyanb/biasbench/experiments/act.txt"
with open(male_word_path, "r") as f:
    for line in f:
        male_words.append(line[:-1])
female_word_path = "/home/pragyanb/biasbench/experiments/cnt.txt"#"/home/pragyanb/biasbench/experiments/cnt.txt"
with open(female_word_path, "r") as f:
    for line in f:
        female_words.append(line[:-1])
female_word_path = "/home/pragyanb/biasbench/experiments/cnt2.txt"
with open(female_word_path, "r") as f:
    for line in f:
        fw2.append(line[:-1])

# %%
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
            lmbda=0, #when using scoring, hyperparamenter for scoring function (increasing it increses debiasing but reduces LM score)
            alpha_ratio=0, #new probs = alpha*debiased_probs + (1-alpha)*vanilla_probs
            softmax_temperature=1, #temperature for vanilla_probs in alpha
            prompt=prefix+prompt[0],
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


def gen_cot(prompt, max_new_tokens=10):
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
            lmbda=0, #when using scoring, hyperparamenter for scoring function (increasing it increses debiasing but reduces LM score)
            alpha_ratio=0, #new probs = alpha*debiased_probs + (1-alpha)*vanilla_probs
            softmax_temperature=1, #temperature for vanilla_probs in alpha
            prompt=cot_prefix+prompt[0],
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

# %%
generations = {}
metrics = {}

# %%
batch_size = 1

for prompt in tqdm(os.listdir("prompts/")):
    generations[prompt] = {}
    metrics[prompt] = {}
    
    wiki_path = "_".join(prompt.split('_')[:-1])
    
    path = os.path.join(f"prompts/{prompt}")
    with open(path, "r") as f:
        prompt_data = json.load(f)
        
    path = os.path.join(f"wikipedia/{wiki_path}_wiki.json")
    with open(path, "r") as f:
        target_data = json.load(f)
    
    for k1, v1 in tqdm(prompt_data.items()):
        generations[prompt][k1] = {}
        metrics[prompt][k1] = {}

        texts = []
        g1 = []
        prompts = []
        impls = []
        prompts2 = []
        g2 = []

        for k2, v2 in v1.items():
            texts += v2

        # generating vanilla outputs
        for i in tqdm(range(0, len(texts), batch_size)):
            if args.method == 'imp':
                g1 += generator(texts[i:i+batch_size])
            elif args.method == 'sd':
                g1 += gen_sd(texts[i:i+batch_size])
            elif args.method == 'cot':
                g1 += gen_cot(texts[i:i+batch_size])
            else:
                g1 += gen_sdb(texts[i:i+batch_size])

        for i in range(len(g1)):
            if i>=len(texts) or i>=len(g1):
                continue
            generations[prompt][k1] = {
                'prompt': texts[i],
                'scoring': g1
            }
        

# %%
with open(f'bold_scoring_new_{args.model_name.split("/")[-1]}_{args.method}_{args.bias_type}.json', 'w') as f:
    json.dump(generations, f)


