import os
import json
from tqdm import tqdm
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sa = SentimentIntensityAnalyzer()

def concatenate_to_sentence(array_of_letters):
    sentence = "".join(array_of_letters)
    return sentence

def get_scores(data):
    scores = {}
    for prompt in tqdm(data.keys()):
        tmp = data[prompt]
        # scores[prompt] = {}
        for group in tmp.keys():
            sents = tmp[group]
            # print(sents)
            scores[group] = {}
            score1_sum = 0
            score2_sum = 0
            total = len(sents.keys())
            # print(sents.keys())
            for key in sents.keys():
                sent = sents[key]
                if key=='scoring':
                    sent = concatenate_to_sentence(sent)
                # print(sents)
                impl_out = sent
                # print(impl_out)
                impl_score = sa.polarity_scores(impl_out)['compound']
                score2_sum += impl_score
            # print(score1_sum, score2_sum, total)
            scores[group] = {'scoring': score2_sum / total} 
    pythia = pd.DataFrame(scores).T
    return pythia      

def get_json_filenames(folder_path):
    json_files = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.json'):
            try:
                with open(filepath, 'r') as file:
                    data = json.load(file)  # Check if the file is valid JSON
                df = get_scores(data)
                trimmed_filename = filename[:-5]  # Remove the last 5 characters (.json)
                json_files.append(trimmed_filename)
                df.to_csv('path')
            except json.JSONDecodeError:
                pass  # File is not valid JSON, skip it
    return json_files

# Example usage:
folder_path = ''
json_filenames = get_json_filenames(folder_path)
print(json_filenames)
