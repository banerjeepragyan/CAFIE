from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
from utils.experiment import ScoringRunner
from utils.stereoset_evaluation import ScoreEvaluator

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

#Specify model, tokenizer, bias types, and other parameters
runner = ScoringRunner(
    intrasentence_model=model,
    tokenizer=tokenizer,
    _do_sdb = False,
    ratio = 0.5,
    scoring_function = "tanh",
    threshold = 0,
    lmbda = 1000, 
    alpha_ratio = 0.99,
    softmax_temperature = 1,
    input_file="",
    model_name_or_path="gpt2",
    batch_size=1,
    is_generative=True,
    is_self_debias=False,
    context_type = 'n', 
    bias_types_to_run = ['gender', 'religion']
)

results = runner()
path = "utils\experiment\outputs\\res.json" #Path to store the results
save_file = open(path, "w")  
json.dump(results, save_file)  
save_file.close() 
score_evaluator = ScoreEvaluator(gold_file_path="data\\test.json", predictions_file_path=path)
overall = score_evaluator.get_overall_results()
print(overall) #Prints the overall results
