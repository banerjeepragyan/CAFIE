import os

from transformers import GPT2Tokenizer, AdamW, get_scheduler, GPT2LMHeadModel
import torch
# from pt_model import GPT2PromptTuningLM
import pandas as pd
import sys
sys.path.append('../')

from biasbench.benchmark.scoring import ScoringRunner
# from scoring_old import ScoringAlgo
# from biasbench.model import models
from biasbench.util import generate_experiment_id, _is_generative
import json
import numpy as np
import random
import os
from stereoset_evaluation import ScoreEvaluator

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

runner = ScoringRunner(
    intrasentence_model=model,
    tokenizer=tokenizer,
    _do_sdb = False,
    ratio = 0.5,
    scoring_function = "tanh", #avg, jpdf, arctan, weight, sigmoid, tanh
    threshold = 0,
    lmbda = 1000,
    alpha_ratio = 0.99,
    softmax_temperature = 1,
    input_file="",
    model_name_or_path="gpt2",
    batch_size=1,
    is_generative=True,
    is_self_debias=False,
    context_type = 'n', #'ab', 'rb', 'fill'
    bias_types_to_run = ['gender', 'race', 'religion', 'profession']
)
results = runner()
print(results)
name = "scoring_chk_multi_countr_chk1"
path = "res.json"
save_file = open(path, "w")  
json.dump(results, save_file)  
save_file.close() 
score_evaluator = ScoreEvaluator(gold_file_path="D:\Documents\IIT-Guwahati\Internships\Adobe\AAAI\CAFIE\data\\test.json", predictions_file_path=path)
overall = score_evaluator.get_overall_results()
print(overall)
