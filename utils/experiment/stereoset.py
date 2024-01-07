from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.experiment import dataloader
from utils.cafie_model import ScoringAlgo

device = "cuda" if torch.cuda.is_available() else "cpu"

class ScoringRunner:
    """Runs StereoSet intrasentence task.

    Notes:
        * We do not evaluate the intersentence task for simplicity. See the actualal
          implementation for intersentence details.
        * Implementation taken from: https://github.com/moinnadeem/StereoSet.
    """

    def __init__(
        self,
        intrasentence_model,
        tokenizer,
        _do_sdb,
        ratio,
        scoring_function,
        threshold,
        lmbda,
        alpha_ratio,
        softmax_temperature,
        model_name_or_path,
        input_file= "data/test.json",
        batch_size=1,
        max_seq_length=128,
        is_generative=False,
        is_self_debias=False,
        bias_type=None,
        context_type = 'ab',
        bias_types_to_run = ['race', 'religion', 'profession', 'gender']
    ):
        """Initializes StereoSet runner.

        Args:
            intrasentence_model: HuggingFace model (e.g., BertForMaskedLM) to evaluate on the
                StereoSet intrasentence task. This can potentially be a debiased model.
            tokenizer: HuggingFace tokenizer (e.g., BertTokenizer) used for pre-processing.
            model_name_or_path: HuggingFace model name (e.g., bert-base-uncased).
            input_file (`str`): Path to the file containing the dataset.
            batch_size (`int`): Batch size used for both the intrasentence and intersentence
                tasks.
            max_seq_length (`int`): Maximum sequence length used for pre-processing. If the
                `batch_size` is 1, there is no maximum.
            is_generative (`bool`): Whether to run the intrasentence task for a generative model or a
                discriminative model.
            is_self_debias (`bool`): Whether we are using a model with self-debiasing or not.
            bias_type (`str`): Bias type for self-debiasing. Determines which prompts are given
                to the model.
        """
        self._intrasentence_model = intrasentence_model
        self._tokenizer = tokenizer
        self._model_name_or_path = model_name_or_path
        self._input_file = input_file
        self._batch_size = batch_size
        self._max_seq_length = None if self._batch_size == 1 else max_seq_length
        self._is_generative = is_generative
        self._is_self_debias = is_self_debias
        # To align with self-debiasing prompt names.
        self._bias_type = "race-color" if bias_type == "race" else bias_type
        self._mask_token = self._tokenizer.mask_token
        self._mask_token_id = self._tokenizer.mask_token_id
        self.do_sdb = _do_sdb
        self.rat = ratio
        self.sf = scoring_function
        self.thres = threshold
        self.lmbd = lmbda
        self.alpha = alpha_ratio
        self.temperature = softmax_temperature
        self.ct = context_type
        self.list_of_bias_types = bias_types_to_run

    def __call__(self):
        bias = {}

        print("Evaluating intrasentence task.")
        intrasentence_bias = self.evaluate_intrasentence()
        bias["intrasentence"] = intrasentence_bias

        return bias

    def evaluate_intrasentence(self):
        sentence_probabilities = self._likelihood_score_generative()
        return sentence_probabilities

    def _likelihood_score_generative(self):
        """Score intrasentence examples using likelihood scoring as proposed by Nadeem et al. for
        generative models (e.g., GPT-2).
        """
        # Use GPU, if available.
        if self._is_self_debias:
            self._intrasentence_model._model.to(device)
        else:
            model = self._intrasentence_model.to(device)

        w1_words = []
        w2_words = []
        w1_words_2 = []
        w3_words = []
        w2_words_2 = []
        w1_word_path = "data\word_lists\list_1.txt"
        with open(w1_word_path, "r") as f:
            for line in f:
                w1_words.append(line[:-1])
        w2_word_path = "data\word_lists\list_2.txt"
        with open(w2_word_path, "r") as f:
            for line in f:
                w2_words.append(line[:-1])
        w3_word_path = "data\word_lists\list_3.txt"
        with open(w3_word_path, "r") as f:
            for line in f:
                w3_words.append(line[:-1])
        w1_word_path2 = "data\word_lists\list_1.txt"
        with open(w1_word_path2, "r") as f:
            for line in f:
                w1_words_2.append(line[:-1])
        w2_word_path2 = "data\word_lists\list_2.txt"
        with open(w2_word_path2, "r") as f:
            for line in f:
                w2_words_2.append(line[:-1])

        # Load the dataset.
        stereoset = dataloader.StereoSet(self._input_file)

        # Assume we are using GPT-2.
        unconditional_start_token = "<|endoftext|>"
        start_token = (
            torch.tensor(self._tokenizer.encode(unconditional_start_token))
            .to(device)
            .unsqueeze(0)
        )

        with torch.no_grad():
            initial_token_probabilities = model(start_token)

        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1
        )
        # Ensure that our batch size is 1 and that our inital token isn't split into subwords.
        assert initial_token_probabilities.shape[0] == 1
        print(initial_token_probabilities.shape)

        clusters = stereoset.get_intrasentence_examples()
        predictions = []

        i=0
        for cluster in tqdm(clusters, leave=False):
            joint_sentence_probability = []
            bt = cluster.bias_type
            context = cluster.context
            if bt in self.list_of_bias_types:
                for sentence in cluster.sentences:
                    i+=1
                    probabilities = {}

                    # Encode the sentence
                    tokens = self._tokenizer.encode(sentence.sentence)
                    with torch.no_grad():
                        joint_sentence_probability = [
                            initial_token_probabilities[0, 0, tokens[0]].item()
                        ]

                        runner = ScoringAlgo(
                            mdl=self._intrasentence_model,
                            model_name_path = self._model_name_or_path,
                            tokenizer=self._tokenizer,
                            _do_sdb=self.do_sdb,
                            ratio=self.rat,
                            scoring_function=self.sf,
                            threshold=self.thres,
                            lmbda=self.lmbd,
                            alpha_ratio=self.alpha,
                            softmax_temperature=self.temperature,
                            prompt=sentence.sentence,
                            context=context,
                            l1=w1_words,
                            act2=w1_words_2,
                            l3 = w3_words,
                            l2=w2_words,
                            cnt2=w2_words_2,
                            sent_len=1,
                            batch_size=1,
                            max_seq_length=128,
                            bias_type=bt,
                            context_type = self.ct,
                        )
                        output, _, _, _ = runner()

                    for idx in range(1, len(tokens)):
                        joint_sentence_probability.append(
                            output[0, idx - 1, tokens[idx]].item()
                        )

                    # Ensure that we have a probability on every token.
                    assert len(tokens) == len(joint_sentence_probability)

                    score = np.sum([np.log2(i) for i in joint_sentence_probability])
                    
                    score /= len(joint_sentence_probability)
                    score = np.power(2, score)
                    probabilities["id"] = sentence.ID
                    probabilities["score"] = score
                    probabilities["label"] = sentence.gold_label

                    predictions.append(probabilities)

        return predictions

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _get_mask_target_tokens(self, s1, s2):
        """Helper function for getting the indices of the target tokens to mask."""
        s1 = s1.tolist()
        if isinstance(s1, int):
            s1 = [s1]
        s2 = s2.tolist()

        idxs = []
        for idx in (i for i, e in enumerate(s2) if e == s1[0]):
            if s2[idx : idx + len(s1)] == s1:
                idxs.append([idx, idx + len(s1) - 1])

        return idxs
