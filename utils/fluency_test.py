import transformers
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
import sys
sys.path.append('../')
from experiments.scoring import ScoringAlgo
# from bias_bench.debias.self_debias.modeling import AutoWrapper
sys.path.append('../../')
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
    default="EleutherAI/pythia-6.9b"
)
args = parser.parse_args()
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
DEBIASING_PREFIXES = [
    "The following text discriminates against people because of their race/color: ",
    "The following text discriminates against people because of their religion: ",
    "The following text discriminates against people because of their gender: "
]

# wrapper = AutoWrapper(model_name=args.model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
tok_xl = transformers.AutoTokenizer.from_pretrained('gpt2-xl')
model_xl = transformers.AutoModelForCausalLM.from_pretrained('gpt2-xl')
model_xl.to(device)

def gen_sdb (text, num_max = 30):
    sss = wrapper.generate_self_debiasing(
        [text], debiasing_prefixes=DEBIASING_PREFIXES, max_length=num_max, num_beams=1, do_sample=True
    )
    # print (text+sss[0])
    return text+sss[0]

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

def get_ppl(gen_sent):
    # model_xl.to(device)
    input_ids = tok_xl(gen_sent, return_tensors='pt').input_ids[0].to(device)
    target_ids = tok_xl(gen_sent, return_tensors='pt').input_ids[0].to(device)
    trg_len = len(input_ids)
    
    with torch.no_grad():
        # import ipdb; ipdb.set_trace()
        outputs = model_xl(input_ids, labels=target_ids)
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

cot_prefix = "The following text is unbiased and not discriminatory: "

v = []
s = []

for prompt in tqdm(prompt_iterator):
    runner = ScoringAlgo(
        mdl=model,
        model_name_path='g',
        tokenizer=tokenizer,
        _do_sdb=False, #add sdb prefix to the sentence
        ratio=0.5, #0.5 for avg
        scoring_function="tanh", #Other options- avg, jpdf, arctan, weight
        threshold=0, #bias threshold for scoring
        lmbda=0, #int(args.lmd), #when using scoring, hyperparamenter for scoring function (increasing it increses debiasing but reduces LM score)
        alpha_ratio=0, #float(args.alp), #new probs = alpha*debiased_probs + (1-alpha)*vanilla_probs
        softmax_temperature=1, #temperature for vanilla_probs in alpha
        prompt=cot_prefix+prompt.text,
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
    output, sc, gen_sent, _ = runner()
    # print(gen_sent)
    # print(get_ppl(gen_sent))
    print(gen_sent)
    # gen_sent = gen_sdb(prompt.text)
    v.append(get_ppl(gen_sent))
    print ( np.mean(v))


print ("=>>>", np.mean(v))
