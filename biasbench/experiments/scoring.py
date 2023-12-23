from collections import defaultdict
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import random
# from transformers import GPT2Tokenizer, AdamW, get_scheduler, GPT2LMHeadModel
import math
# seed = 0
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = "cuda" if torch.cuda.is_available() else "cpu"

class ScoringAlgo:

    def __init__(
        self,
        mdl,
        model_name_path,
        tokenizer,
        _do_sdb,
        ratio,
        scoring_function,
        threshold,
        lmbda,
        alpha_ratio,
        softmax_temperature,
        prompt,
        context,
        act1,
        cnt1,
        ct2,
        act2=[],
        cnt2=[],
        sent_len=10,
        bias_type="none",
        batch_size=1,
        max_seq_length=128,
        gamma=1,
        words_to_ignore = [],
        context_type = 'ab',
    ):
        self._intrasentence_model = mdl
        self.model_name = model_name_path
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._max_seq_length = None if self._batch_size == 1 else max_seq_length
        # To align with self-debiasing prompt names.
        # self._bias_type = "race-color" if bias_type == "race" else bias_type
        # self._mask_token = self._tokenizer.mask_token
        # self._mask_token_id = self._tokenizer.mask_token_id
        self._bias_type = bias_type
        self.do_sdb = _do_sdb
        self.rat = ratio
        self.sf = scoring_function
        self.thres = threshold
        self.lmbd = lmbda
        self.alpha = alpha_ratio
        self.temperature = softmax_temperature
        self.prmpt = prompt
        self.cntxt = context
        self.male_words = act1
        self.male_words_2 = act2
        self.female_words = cnt1
        self.female_wrds2 = ct2
        self.female_words_2 = cnt2
        self.sl = sent_len
        self.gma = gamma
        self.words_ignore = words_to_ignore
        self.ct = context_type

    def __call__(self):
        with torch.no_grad():
            self._intrasentence_model.to(device)
            output, sc, sent, da = self.generate_sentences (self.prmpt, self.sl, self.male_words, self.female_words, self.female_wrds2, self.male_words_2, self.female_words_2, self.cntxt, self._bias_type)
        return output, sc, sent, da

    def find_reqd_indices(self, arr1, arr2):
        j=0
        matching_indices1 = []
        matching_indices2 = []
        for i in range(len(arr1)):
            c = 0
            while j < len(arr2) and arr2[j] not in arr1:
                if c==0:
                    matching_indices1.append(i)
                    matching_indices2.append(j)
                c+=1
                j+=1
            if j < len(arr2) and arr2[j]==arr1[i]:
                matching_indices1.append(i)
                matching_indices2.append(j)
                j+=1
        m1 = np.array(matching_indices1)
        m2 = np.array(matching_indices2)
        m1 = np.sort(len(arr1)-1-m1)
        m2 = np.sort(len(arr2)-1-m2)
        return m1.tolist(), m2.tolist()
        # return matching_indices1, matching_indices2

    def generate_probabilities (self, input_ids, past, attention_mask, use_cache): 
        # print(device)
        if self.model_name=='b': 
            print()
            outputs = self._intrasentence_model(input_ids.to(device)).logits
            ll = outputs[0, -1, :].to(device)
            probs = torch.nn.functional.softmax(ll/self.temperature, dim=-1).to(device)
            return probs, outputs, torch.nn.functional.softmax(outputs/self.temperature, dim=-1), outputs
        else:
            outputs = self._intrasentence_model(input_ids.to(device))
            ll = outputs[0][0, -1, :].to(device)
            probs = torch.nn.functional.softmax(ll/self.temperature, dim=-1).to(device)
            return probs, outputs[1], torch.nn.functional.softmax(outputs[0]/self.temperature, dim=-1), outputs[0]
        # return probs, outputs[1], outputs[0], outputs[0].clone().to(device)

    def calculate_new_pdf (self, probs_origin, scores, scres2, avg, thres, sf, lmbd, iil):
        dont = torch.zeros(scores.shape).to(device)
        a = scores < thres
        b = scores > -thres
        c = a & b
        scores = torch.where(c, dont, scores)
        scores[:, :, iil] = 0
        a = scres2 < thres
        b = scres2 > -thres
        c = a & b
        scres2 = torch.where(c, dont, scres2)
        scres2[:, :, iil] = 0
        d1 = torch.exp(torch.linalg.vector_norm(scores))
        d2 = torch.exp(torch.linalg.vector_norm(scres2))
        
        if sf=="arctan":
            wt1 = (2/math.pi)*torch.atan(-lmbd*scores)+1
            wt2 = (2/math.pi)*torch.atan(-lmbd*scres2)+1
        elif sf=="weight":
            wt1 = 1 - torch.nn.functional.softmax(scores/self.temperature, dim=-1)
            wt2 = 1 - torch.nn.functional.softmax(scres2/self.temperature, dim=-1)
        else:
            wt1 = (self.gma*torch.tanh(-lmbd*scores))+1
            wt2 = (self.gma*torch.tanh(-lmbd*scres2))+1

            
        penalised_probs1 = torch.mul(wt1, probs_origin)
        penalised_probs2 = torch.mul(wt2, probs_origin)


        # if d1 > d2:
        #     probs_new = penalised_probs1
        # else:
        #     probs_new = penalised_probs2

        probs_new = (d1/(d1+d2))*penalised_probs1 + (d2/(d1+d2))*penalised_probs2
        return probs_new, (d1+d2)/2.

    def calculate_scores_1 (self, probs_origin, probs_countr, probs_ncontr, r):
        scores = probs_origin-probs_countr
        scres2 = probs_origin-probs_ncontr
        return scores, (r*probs_origin)+((1-r)*probs_countr), scres2

    def topk_process(self, outputs, k):
        for i in range(len(outputs[0])):
            logits = outputs[0][i]
            indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
            logits[indices_to_remove] = 0
            outputs[0][i] = logits
        return outputs

    def remove_bad(self, act, sdb, l):
        s_a = sdb[:, -l:, :]
        delta = act - s_a
        dlt = torch.where(delta>0, torch.ones_like(delta), torch.exp(100*delta))
        pos_mask = delta > 0
        delta[pos_mask] = 1
        delta[~pos_mask] = torch.exp(50*delta[~pos_mask])
        p_hat = torch.mul(delta, act)
        p_sdb = torch.nn.functional.softmax(p_hat/self.temperature, dim=-1)
        return p_sdb
    
    def beam_search_decoder(self, data, k):
        sequences = [[list(), 0.0]]
        for row in data:
            all_candidates = list()
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - math.log(row[j])]
                    all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:k]
        return sequences

    
    def create_countr (self, male_words, female_words, new_cnt_words, prompt, bt):
        prompt_actual = prompt
        prompt_actual_array = prompt_actual.split()
        prompt_countr_array = []
        prompt_neutrl_array = []
        prompt_ncontr_array = []
        rel_array = []
        for word in prompt_actual_array:
            if word in male_words:
                # if bt == "gender":
                prompt_countr_array.append(female_words[male_words.index(word)])
                prompt_ncontr_array.append(new_cnt_words[male_words.index(word)])
                # else:
                #     prompt_countr_array.append("non-"+word)
                rel_array.append(word)
                prompt_neutrl_array.append("they")
            elif word in female_words:
                # if bt == "gender":
                prompt_countr_array.append(male_words[female_words.index(word)])
                prompt_ncontr_array.append(new_cnt_words[female_words.index(word)])
                # else:
                #     prompt_countr_array.append("non-"+word)
                # prompt_countr_array.append(male_words[female_words.index(word)])
                rel_array.append(word)
                prompt_neutrl_array.append("they")
            elif word in new_cnt_words:
                # if bt == "gender":
                prompt_countr_array.append(male_words[new_cnt_words.index(word)])
                prompt_ncontr_array.append(female_words[new_cnt_words.index(word)])
                # else:
                #     prompt_countr_array.append("non-"+word)
                # prompt_countr_array.append(male_words[female_words.index(word)])
                rel_array.append(word)
                prompt_neutrl_array.append("they")
            else:
                prompt_countr_array.append(word)
                prompt_ncontr_array.append(word)
                prompt_neutrl_array.append(word)
        prompt_countr = ""
        for word in prompt_countr_array:
            prompt_countr += word + " "
        prompt_countr = prompt_countr[:-1]
        prompt_ncontr = ""
        for word in prompt_ncontr_array:
            prompt_ncontr += word + " "
        prompt_ncontr = prompt_ncontr[:-1]
        prompt_neutrl = ""
        for word in prompt_neutrl_array:
            prompt_neutrl += word + " "
        prompt_neutrl = prompt_neutrl[:-1]
        rel_words = ""
        for word in rel_array:
            rel_words += word + " "
        rel_words = rel_words[:-1]
        return prompt_actual, prompt_countr, prompt_ncontr, prompt_neutrl, rel_words
    
    def fill_blank_biased(self, context):
        r = re.split('(BLANK)', context)
        if r[0]=="":
            contxt = " "
        else:
            contxt = r[0]
        input_ids = self._tokenizer.encode(contxt, return_tensors="pt")
        input_list = input_ids.cpu().detach().numpy().tolist()[0]
        input_lists = [input_list]
        input_ids = torch.LongTensor(input_lists)   
        attention_mask = input_ids.new_ones(input_ids.shape)
        outputs = self._intrasentence_model(input_ids.to(device))
        ll = torch.argmax(torch.nn.functional.softmax(outputs[0][0, -1, :].to(device))).to(device)
        word = self._tokenizer.decode(ll)
        return contxt + word + r[-1]

    def generate_sentences (self, prompt, sent_len, male_words, female_words, new_cnt_words, male_words_2, female_words_2, context, bt):
        sdb_prompt = "The following sentence contains very hateful, aggressive, disrespectful, toxic language- "

        if self.ct == 'ab':

            r = re.split('(BLANK)', context)
            context = r[-1]

        elif self.ct == 'i':

            context = "Fill in the BLANK- " + context

        elif self.ct == 'fill':

            context = self.fill_blank_biased(context)

        elif self.ct == 'rb':

            context = context.replace("BLANK", "")
        else:

            context = ""

        prompt_origin, prompt_countr, prompt_ncontr, prompt_neutrl, rel_words = self.create_countr (male_words, female_words, new_cnt_words, prompt, bt)
        contxt_origin, contxt_countr, contxt_ncontr, contxt_neutrl, rel_cntxt = self.create_countr (male_words, female_words, new_cnt_words, context, bt)

        # print(prompt_origin)

        # print(prompt_countr)

        # print(prompt_ncontr)

        # if self.do_sdb:
        #     prompt_origin = sdb_prompt + prompt_origin
        #     prompt_countr = sdb_prompt + prompt_countr

        prompt_actual = prompt_origin

        prompt_countr_sdb = sdb_prompt + prompt_countr
        # prompt_origin_sdb = sdb_prompt + prompt_origin

        input_ids_origin = self._tokenizer.encode(prompt_origin, return_tensors="pt")

        # print("IDS SC ", input_ids_origin.shape)
        input_list_origin = input_ids_origin.cpu().detach().numpy().tolist()[0]
        input_lists_origin = [input_list_origin]
        input_ids_origin = torch.LongTensor(input_lists_origin) 
        attention_mask_origin = input_ids_origin.new_ones(input_ids_origin.shape)

        input_ids_actual = self._tokenizer.encode(prompt_actual, return_tensors="pt")
        input_list_actual = input_ids_actual.cpu().detach().numpy().tolist()[0]
        input_lists_actual = [input_list_actual]
        input_ids_actual = torch.LongTensor(input_lists_actual) 
        attention_mask_actual = input_ids_actual.new_ones(input_ids_actual.shape)

        _,l = input_ids_actual.shape

        input_ids_countr = self._tokenizer.encode(prompt_countr, return_tensors="pt")
        input_list_countr = input_ids_countr.cpu().detach().numpy().tolist()[0]
        input_lists_countr = [input_list_countr]
        input_ids_countr = torch.LongTensor(input_lists_countr)   
        attention_mask_countr = input_ids_countr.new_ones(input_ids_countr.shape)

        input_ids_ncontr = self._tokenizer.encode(prompt_ncontr, return_tensors="pt")
        input_list_ncontr = input_ids_ncontr.cpu().detach().numpy().tolist()[0]
        input_lists_ncontr = [input_list_ncontr]
        input_ids_ncontr = torch.LongTensor(input_lists_ncontr)   
        attention_mask_ncontr = input_ids_ncontr.new_ones(input_ids_ncontr.shape)

        # input_ids_cntr_2 = self._tokenizer.encode(prompt_cntr_2, return_tensors="pt")
        # input_list_cntr_2 = input_ids_cntr_2.cpu().detach().numpy().tolist()[0]
        # input_lists_cntr_2 = [input_list_cntr_2]
        # input_ids_cntr_2 = torch.LongTensor(input_lists_cntr_2)   
        # attention_mask_cntr_2 = input_ids_cntr_2.new_ones(input_ids_cntr_2.shape)

        input_ids_neutrl = self._tokenizer.encode(prompt_neutrl, return_tensors="pt")
        input_list_neutrl = input_ids_neutrl.cpu().detach().numpy().tolist()[0]
        input_lists_neutrl = [input_list_neutrl]
        input_ids_neutrl = torch.LongTensor(input_lists_neutrl)   
        attention_mask_neutrl = input_ids_neutrl.new_ones(input_ids_neutrl.shape)

        input_ids_origin_c = self._tokenizer.encode(contxt_origin, return_tensors="pt")
        input_list_origin_c = input_ids_origin_c.cpu().detach().numpy().tolist()[0]
        input_lists_origin_c = [input_list_origin_c]
        input_ids_origin_c = torch.LongTensor(input_lists_origin_c) 
        attention_mask_origin_c = input_ids_origin_c.new_ones(input_ids_origin_c.shape)

        input_ids_countr_c = self._tokenizer.encode(contxt_countr, return_tensors="pt")
        input_list_countr_c = input_ids_countr_c.cpu().detach().numpy().tolist()[0]
        input_lists_countr_c = [input_list_countr_c]
        input_ids_countr_c = torch.LongTensor(input_lists_countr_c)   
        attention_mask_countr_c = input_ids_countr_c.new_ones(input_ids_countr_c.shape)

        input_ids_ncontr_c = self._tokenizer.encode(contxt_ncontr, return_tensors="pt")
        input_list_ncontr_c = input_ids_ncontr_c.cpu().detach().numpy().tolist()[0]
        input_lists_ncontr_c = [input_list_ncontr_c]
        input_ids_ncontr_c = torch.LongTensor(input_lists_ncontr_c)   
        attention_mask_ncontr_c = input_ids_ncontr_c.new_ones(input_ids_ncontr_c.shape)

        # input_ids_ac_sdb = self._tokenizer.encode(prompt_origin_sdb, return_tensors="pt")
        # input_list_ac_sdb = input_ids_ac_sdb.cpu().detach().numpy().tolist()[0]
        # input_lists_ac_sdb = [input_list_ac_sdb]
        # input_ids_ac_sdb = torch.LongTensor(input_lists_ac_sdb)   
        # attention_mask_ac_sdb = input_ids_ac_sdb.new_ones(input_ids_ac_sdb.shape)

        # input_ids_cn_sdb = self._tokenizer.encode(prompt_countr_sdb, return_tensors="pt")
        # input_list_cn_sdb = input_ids_cn_sdb.cpu().detach().numpy().tolist()[0]
        # input_lists_cn_sdb = [input_list_cn_sdb]
        # input_ids_cn_sdb = torch.LongTensor(input_lists_cn_sdb)   
        # attention_mask_cn_sdb = input_ids_cn_sdb.new_ones(input_ids_cn_sdb.shape)

        past_origin, past_countr, past_ncontr, past_actual, past_cntr_2, past_neutrl, past_ac_sdb, past_cn_sdb, use_cache = None, None, None, None, None, None, None, None, True
        sent = prompt_origin
        final_probs_new = []

        outputs = None

        reqd_indices_origin, reqd_indices_countr = self.find_reqd_indices(torch.flip(input_ids_origin[0], dims=[-1]), torch.flip(input_ids_countr[0], dims=[-1]))
        reqd_in_cntx_origin, reqd_in_cntx_countr = self.find_reqd_indices(torch.flip(input_ids_origin_c[0], dims=[-1]), torch.flip(input_ids_countr_c[0], dims=[-1]))
        reqd_indices_ncontr = reqd_indices_countr
        reqd_in_cntx_ncontr = reqd_in_cntx_countr

        if self.ct != 'n':

            input_ids_origin = torch.cat([input_ids_origin_c, input_ids_origin], dim=-1)
            input_ids_countr = torch.cat([input_ids_countr_c, input_ids_countr], dim=-1)
            input_ids_ncontr = torch.cat([input_ids_ncontr_c, input_ids_ncontr], dim=-1)

        ignore_list = []
        for word in self.words_ignore:
            ignore_list.append(' ' + word)
        ignore_ids_list = []
        for word in ignore_list:
            w_ids = self._tokenizer.encode(word)
            for w in w_ids:
                ignore_ids_list.append(w)
 
        if self.do_sdb:
            sdb_ids = self._tokenizer.encode(sdb_prompt, return_tensors="pt")
            input_ids_origin = torch.cat([sdb_ids, input_ids_origin], dim=-1)
            # input_ids_countr = torch.cat([sdb_ids, input_ids_countr], dim=-1)

        # print(self._tokenizer.decode(input_ids_origin.tolist()[0]))

        reqd_indices_origin = [x + len(input_ids_origin_c[0]) for x in reqd_indices_origin]
        reqd_indices_countr = [x + len(input_ids_countr_c[0]) for x in reqd_indices_countr]
        reqd_indices_ncontr = [x + len(input_ids_ncontr_c[0]) for x in reqd_indices_ncontr]

        reqd_indices_origin = reqd_in_cntx_origin + reqd_indices_origin
        reqd_indices_countr = reqd_in_cntx_countr + reqd_indices_countr
        reqd_indices_ncontr = reqd_in_cntx_ncontr + reqd_indices_ncontr

        reqd_indices_origin.append(len(input_ids_origin[0])-1)
        reqd_indices_countr.append(len(input_ids_countr[0])-1)
        reqd_indices_ncontr.append(len(input_ids_ncontr[0])-1)
        # reqd_indices_origin, reqd_indices_countr = self.find_reqd_indices(input_ids_origin[0], input_ids_countr[0])

        # reqd_indices_ac_sdb, reqd_indices_cn_sdb = self.find_matching_indices(input_ids_ac_sdb[0], input_ids_cn_sdb[0])
        # reqd_indices_actl_2, reqd_indices_cntr_2 = self.find_matching_indices(input_ids_origin[0], input_ids_cntr_2[0])

        # reqd_indices_actl_2, reqd_indices_cntr_2 = self.find_reqd_indices(torch.flip(input_ids_origin[0], dims=[-1]), torch.flip(input_ids_cntr_2[0], dims=[-1]))
        # reqd_indices_actl_2, reqd_indices_cntr_2 = self.find_reqd_indices(input_ids_origin[0], input_ids_cntr_2[0])

        len_origin = len(input_ids_origin[0])
        len_countr = len(input_ids_countr[0])
        len_ncontr = len(input_ids_ncontr[0])
        # len_ac_sdb = len(input_ids_ac_sdb[0])
        # len_cn_sdb = len(input_ids_cn_sdb[0])

        indices_skipped_origin = []
        indices_skipped_countr = []
        indices_skipped_ncontr = []
        indices_skipped_cntr_2 = []

        for i in range(len(input_ids_origin[0])):
            if i not in reqd_indices_origin:
                indices_skipped_origin.append(i)

        for i in range(len(input_ids_countr[0])):
            if i not in reqd_indices_countr:
                indices_skipped_countr.append(i)

        for i in range(len(input_ids_ncontr[0])):
            if i not in reqd_indices_ncontr:
                indices_skipped_ncontr.append(i)

        # for i in range(len(input_ids_cntr_2[0])):
        #     if i not in reqd_indices_cntr_2:
        #         # print(i)
        #         indices_skipped_cntr_2.append(i)

        jsp = []

        for i in range(sent_len):
            probs_origin, past_origin, outputs_origin, ro_origin = self.generate_probabilities (input_ids_origin, past_origin, attention_mask_origin, use_cache)
            probs_countr, past_countr, outputs_countr, ro_countr = self.generate_probabilities (input_ids_countr, past_countr, attention_mask_countr, use_cache)
            # print(outputs_origin.shape)
            probs_ncontr, past_ncontr, outputs_ncontr, ro_ncontr = self.generate_probabilities (input_ids_ncontr, past_ncontr, attention_mask_ncontr, use_cache)
            
            # probs_actual, past_actual, outputs_actual, ro_actual = self.generate_probabilities (input_ids_actual, past_actual, attention_mask_actual, use_cache)
            # probs_cntr_2, past_cntr_2, outputs_cntr_2, ro_cntr_2 = self.generate_probabilities (input_ids_cntr_2, past_cntr_2, attention_mask_cntr_2, use_cache)
            # probs_neutrl, past_neutrl, outputs_neutrl = self.generate_probabilities (input_ids_neutrl, past_neutrl, attention_mask_neutrl, use_cache)
            # probs_ac_sdb, past_ac_sdb, outputs_ac_sdb = self.generate_probabilities (input_ids_ac_sdb, past_ac_sdb, attention_mask_ac_sdb, use_cache)
            # probs_cn_sdb, past_cn_sdb, outputs_cn_sdb = self.generate_probabilities (input_ids_cn_sdb, past_cn_sdb, attention_mask_cn_sdb, use_cache)

            logit_id_probable = torch.argmax(outputs_origin[0][-1]).to(device)

            outputs_origin_copy = outputs_origin.clone().to(device)
            outputs_origin = outputs_origin.to(device)
            # outputs_actl_2 = outputs_origin.clone().to(device)
            # outputs_origin = self.topk_process(outputs_origin, 10).to(device)
            # outputs_origin_trimmed = torch.index_select(outputs_origin.to(device), 1, torch.Tensor(reqd_indices_origin).int().to(device))
            # outputs_countr_trimmed = torch.index_select(outputs_countr.to(device), 1, torch.Tensor(reqd_indices_countr).int().to(device))

            # p1 = self.remove_bad (outputs_origin, outputs_ac_sdb, len_origin)
            # p2 = self.remove_bad (outputs_countr, outputs_cn_sdb, len_countr)

            # outputs_origin = p1
            # outputs_countr = p2

            outputs_origin_trimmed = torch.index_select(outputs_origin.to(device), 1, torch.Tensor(reqd_indices_origin).int().to(device))
            outputs_countr_trimmed = torch.index_select(outputs_countr.to(device), 1, torch.Tensor(reqd_indices_countr).int().to(device))
            # outputs_ncontr_trimmed = torch.index_select(outputs_ncontr.to(device), 1, torch.Tensor(reqd_indices_ncontr).int().to(device))
            outputs_ncontr_trimmed = outputs_countr_trimmed
            ids_origin_trimmed = torch.index_select(input_ids_origin.to(device), -1, torch.Tensor(reqd_indices_origin).int().to(device)).to(device)
            scores, avg, scres2 = self.calculate_scores_1 (outputs_origin_trimmed, outputs_countr_trimmed, outputs_ncontr_trimmed, self.rat)

            for idx in range(1,len(ids_origin_trimmed[0])):
                jsp.append(scores[0, idx-1, ids_origin_trimmed[0][idx]].item())


            # outputs_actl_2_trimmed = torch.index_select(outputs_actl_2.to(device), 1, torch.Tensor(reqd_indices_actl_2).int().to(device))
            # outputs_cntr_2_trimmed = torch.index_select(outputs_cntr_2.to(device), 1, torch.Tensor(reqd_indices_cntr_2).int().to(device))
            # scrs_2 = self.calculate_scores_1 (outputs_actl_2_trimmed, outputs_cntr_2_trimmed)

            # scores = (scrs_1[:, -x:, :]+scrs_2[:, -x:, :])/2
            # scores = scrs_1

            # outputs_origin_trimmed = self.topk_process(outputs_origin_trimmed, 10)
            # scores = calculate_scores_2 (outputs_origin, outputs_countr, outputs_neutrl)

            # if a < b:
            #     probs_new = self.calculate_new_pdf(outputs_origin_trimmed, scores)
            #     outputs_origin[0][torch.Tensor(reqd_indices_origin).long()] = probs_new
            # else:
            #     probs_new = self.calculate_new_pdf(outputs_actl_2_trimmed, scores)
            #     outputs_origin[0][torch.Tensor(reqd_indices_actl_2).long()] = probs_new

            da = None

            if self.sf=="avg":
                probs_new = avg
            elif self.sf=="jpdf":
                probs_new = torch.mul(outputs_origin_trimmed,outputs_countr_trimmed)
            else:
                probs_new, da = self.calculate_new_pdf(outputs_origin_trimmed, scores, scres2, avg, self.thres, self.sf, self.lmbd, ignore_ids_list)

            outputs_origin[0][torch.Tensor(reqd_indices_origin).long()] = probs_new.to(device)   #.to(dtype=torch.bfloat16)

            # logit_id = torch.argmin(scores)
            # outputs_origin = torch.nn.functional.softmax(outputs_origin, dim=-1)
            # outputs_origin_copy = torch.nn.functional.softmax(outputs_origin_copy, dim=-1)            

            # outputs_actual = alpha*outputs_origin[:,-l:,:] + (1-alpha)*torch.nn.functional.softmax(ro_origin[:, -l:, :]/temperature, dim=-1)  #outputs_origin_copy
            # outputs_actual = self.alpha*torch.nn.functional.softmax(outputs_origin[:,-l:,:], dim=-1) + (1-self.alpha)*torch.nn.functional.softmax(ro_origin[:, -l:, :]/temperature, dim=-1)  #outputs_origin_copy
            # if da != 0 or da != None:
            #     aa = 1.65/da
            # else:
            #     aa = self.alpha

            aa = self.alpha


            outputs_actual = self.alpha*outputs_origin[:,-l:,:] + (1-self.alpha)*torch.nn.functional.softmax(ro_origin[:, -l:, :]/self.temperature, dim=-1)  #outputs_origin_copy

            # outputs_origin = torch.nn.functional.softmax(outputs_origin, dim=-1)

            # import ipdb; ipdb.set_trace()
            # logit_id = self.beam_search_decoder(outputs_actual[0],5)[0][0][0]
            # logit_id = torch.argmax(outputs_actual[0][-1]).to(device)
            logit_id = torch.multinomial(outputs_actual[0][-1], num_samples=1).to(device)[0]
            word = self._tokenizer.decode(logit_id)
            sent += word

            input_ids_origin = torch.cat([input_ids_origin.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            attention_mask_origin = torch.cat([attention_mask_origin, attention_mask_origin.new_ones((attention_mask_origin.shape[0], 1))], dim=-1)
            input_ids_countr = torch.cat([input_ids_countr.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            attention_mask_countr = torch.cat([attention_mask_countr, attention_mask_countr.new_ones((attention_mask_countr.shape[0], 1))], dim=-1)
            input_ids_ncontr = torch.cat([input_ids_ncontr.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            attention_mask_ncontr = torch.cat([attention_mask_ncontr, attention_mask_countr.new_ones((attention_mask_ncontr.shape[0], 1))], dim=-1)
            # input_ids_actual = torch.cat([input_ids_actual.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            # attention_mask_actual = torch.cat([attention_mask_actual, attention_mask_actual.new_ones((attention_mask_actual.shape[0], 1))], dim=-1)
            # input_ids_neutrl = torch.cat([input_ids_neutrl.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            # attention_mask_neutrl = torch.cat([attention_mask_neutrl, attention_mask_neutrl.new_ones((attention_mask_neutrl.shape[0], 1))], dim=-1)
            # input_ids_ac_sdb = torch.cat([input_ids_ac_sdb.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            # attention_mask_ac_sdb = torch.cat([attention_mask_ac_sdb, attention_mask_ac_sdb.new_ones((attention_mask_ac_sdb.shape[0], 1))], dim=-1)
            # input_ids_cn_sdb = torch.cat([input_ids_cn_sdb.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            # attention_mask_cn_sdb = torch.cat([attention_mask_cn_sdb, attention_mask_cn_sdb.new_ones((attention_mask_cn_sdb.shape[0], 1))], dim=-1)

            reqd_indices_origin.append(i+len_origin)
            reqd_indices_countr.append(i+len_countr)
            reqd_indices_ncontr.append(i+len_ncontr)

            # outputs_origin = torch.nn.functional.softmax(outputs_origin, dim=-1)
            # outputs = torch.cat((outputs_origin[0, :-1, :], probs_new.unsqueeze(0)), dim=0).unsqueeze(0)

        return outputs_actual, np.mean(jsp), sent, da