import transformers
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
import sys
sys.path.append('../../')
from biasbench.experiments.scoring_b import ScoringAlgo
from self_debiasing.io_utils import load_prompts
from tqdm import tqdm
device = "cuda"
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Blah")
parser.add_argument(
    "--alp",
    default=0.0
)
parser.add_argument(
    "--lmd",
    default=100
)
parser.add_argument(
    "--model_name",
    default="openlm-research/open_llama_13b"
)
args = parser.parse_args()
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_name, decode_with_prefix_space=True)
# tok_xl = transformers.AutoTokenizer.from_pretrained('gpt2-xl')
# model_xl = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')
# model_xl.to(device)

def get_ppl(gen_sent):
    # model_xl.to(device)
    input_ids = tokenizer(gen_sent, return_tensors='pt').input_ids[0].to(device)
    target_ids = tokenizer(gen_sent, return_tensors='pt').input_ids[0].to(device)
    trg_len = len(input_ids)
    
    with torch.no_grad():
        # import ipdb; ipdb.set_trace()
        outputs = model(input_ids.unsqueeze(dim=0), labels=target_ids.unsqueeze(dim=0))
        lm_logits = outputs[1]
        lls = []
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        log_likelihood = loss * trg_len
        lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / trg_len)
        x = ppl.cpu().tolist()
        # print(x)
        return x

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

# prompt_iterator = ["Two black boys start a"]
prompts = load_prompts("", challenging_only=True)
prompt_iterator = tqdm(prompts, desc="Prompts")

cot_prefix = "Generate some text that treats all groups of people the same and does not discriminate between them. Lets think step by step: "

v = []
s = []

for prompt in prompt_iterator:
    # print(1)
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
        prompt=prompt.text,
        context="",
        act1=male_words,
        cnt1=female_words,
        ct2=fw2,
        sent_len=30,
        batch_size=1,
        max_seq_length=128,
        bias_type=None,
        words_to_ignore = []
    )
    # print(2)
    output, sc, gen_sent, _ = runner()
    # print(gen_sent)
    # print(get_ppl(gen_sent))
    # print(gen_sent)
    x = get_ppl(gen_sent)
    # print(x)
    v.append(x)
    # print ( np.mean(v))


print ("=>>>", np.mean(v))
