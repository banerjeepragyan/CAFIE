import re
import numpy as np
import torch
from tqdm.notebook import tqdm
import math

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
        l1,
        l2,
        l3,
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
        self.w1_words = l1
        self.w1_words_2 = act2
        self.w2_words = l2
        self.w3_wrds2 = l3
        self.w2_words_2 = cnt2
        self.sl = sent_len
        self.gma = gamma
        self.words_ignore = words_to_ignore
        self.ct = context_type

    def __call__(self):
        with torch.no_grad():
            self._intrasentence_model.to(device)
            output, sc, sent, da = self.generate_sentences (self.prmpt, self.sl, self.w1_words, self.w2_words, self.w3_wrds2, self.w1_words_2, self.w2_words_2, self.cntxt, self._bias_type)
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

    def generate_probabilities (self, input_ids, past, attention_mask, use_cache): 
        outputs = self._intrasentence_model(input_ids.to(device))
        ll = outputs[0][0, -1, :].to(device)
        probs = torch.nn.functional.softmax(ll/self.temperature, dim=-1).to(device)
        return probs, outputs[1], torch.nn.functional.softmax(outputs[0]/self.temperature, dim=-1), outputs[0]

    def calculate_new_pdf (self, probs_w1, scores, scres2, avg, thres, sf, lmbd, iil):
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

        wt1 = (self.gma*torch.tanh(-lmbd*scores))+1
        wt2 = (self.gma*torch.tanh(-lmbd*scres2))+1
            
        penalised_probs1 = torch.mul(wt1, probs_w1)
        penalised_probs2 = torch.mul(wt2, probs_w1)

        probs_new = (d1/(d1+d2))*penalised_probs1 + (d2/(d1+d2))*penalised_probs2
        return probs_new, (d1+d2)/2.

    def calculate_scores_1 (self, probs_w1, probs_w2, probs_w3, r):
        scores = probs_w1-probs_w2
        scres2 = probs_w1-probs_w3
        return scores, (r*probs_w1)+((1-r)*probs_w2), scres2

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
 
    def create_w2 (self, w1_words, w2_words, w3_words, prompt, bt):
        prompt_actual = prompt
        prompt_actual_array = prompt_actual.split()
        prompt_w2_array = []
        prompt_neutrl_array = []
        prompt_w3_array = []
        rel_array = []
        for word in prompt_actual_array:
            if word in w1_words:
                prompt_w2_array.append(w2_words[w1_words.index(word)])
                prompt_w3_array.append(w3_words[w1_words.index(word)])
                rel_array.append(word)
                prompt_neutrl_array.append("they")
            elif word in w2_words:
                prompt_w2_array.append(w1_words[w2_words.index(word)])
                prompt_w3_array.append(w3_words[w2_words.index(word)])
                rel_array.append(word)
                prompt_neutrl_array.append("they")
            elif word in w3_words:
                prompt_w2_array.append(w1_words[w3_words.index(word)])
                prompt_w3_array.append(w2_words[w3_words.index(word)])
                rel_array.append(word)
                prompt_neutrl_array.append("they")
            else:
                prompt_w2_array.append(word)
                prompt_w3_array.append(word)
                prompt_neutrl_array.append(word)
        prompt_w2 = ""
        for word in prompt_w2_array:
            prompt_w2 += word + " "
        prompt_w2 = prompt_w2[:-1]
        prompt_w3 = ""
        for word in prompt_w3_array:
            prompt_w3 += word + " "
        prompt_w3 = prompt_w3[:-1]
        prompt_neutrl = ""
        for word in prompt_neutrl_array:
            prompt_neutrl += word + " "
        prompt_neutrl = prompt_neutrl[:-1]
        rel_words = ""
        for word in rel_array:
            rel_words += word + " "
        rel_words = rel_words[:-1]
        return prompt_actual, prompt_w2, prompt_w3, prompt_neutrl, rel_words
    
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
        outputs = self._intrasentence_model(input_ids.to(device))
        ll = torch.argmax(torch.nn.functional.softmax(outputs[0][0, -1, :].to(device))).to(device)
        word = self._tokenizer.decode(ll)
        return contxt + word + r[-1]

    def generate_sentences (self, prompt, sent_len, w1_words, w2_words, w3_words, w1_words_2, w2_words_2, context, bt):
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
        prompt_w1, prompt_w2, prompt_w3, prompt_neutrl, rel_words = self.create_w2 (w1_words, w2_words, w3_words, prompt, bt)
        contxt_w1, contxt_w2, contxt_w3, contxt_neutrl, rel_cntxt = self.create_w2 (w1_words, w2_words, w3_words, context, bt)

        prompt_actual = prompt_w1
        input_ids_w1 = self._tokenizer.encode(prompt_w1, return_tensors="pt")

        input_list_w1 = input_ids_w1.cpu().detach().numpy().tolist()[0]
        input_lists_w1 = [input_list_w1]
        input_ids_w1 = torch.LongTensor(input_lists_w1) 
        attention_mask_w1 = input_ids_w1.new_ones(input_ids_w1.shape)

        input_ids_actual = self._tokenizer.encode(prompt_actual, return_tensors="pt")
        input_list_actual = input_ids_actual.cpu().detach().numpy().tolist()[0]
        input_lists_actual = [input_list_actual]
        input_ids_actual = torch.LongTensor(input_lists_actual) 

        _,l = input_ids_actual.shape

        input_ids_w2 = self._tokenizer.encode(prompt_w2, return_tensors="pt")
        input_list_w2 = input_ids_w2.cpu().detach().numpy().tolist()[0]
        input_lists_w2 = [input_list_w2]
        input_ids_w2 = torch.LongTensor(input_lists_w2)   
        attention_mask_w2 = input_ids_w2.new_ones(input_ids_w2.shape)

        input_ids_w3 = self._tokenizer.encode(prompt_w3, return_tensors="pt")
        input_list_w3 = input_ids_w3.cpu().detach().numpy().tolist()[0]
        input_lists_w3 = [input_list_w3]
        input_ids_w3 = torch.LongTensor(input_lists_w3)   
        attention_mask_w3 = input_ids_w3.new_ones(input_ids_w3.shape)

        input_ids_neutrl = self._tokenizer.encode(prompt_neutrl, return_tensors="pt")
        input_list_neutrl = input_ids_neutrl.cpu().detach().numpy().tolist()[0]
        input_lists_neutrl = [input_list_neutrl]
        input_ids_neutrl = torch.LongTensor(input_lists_neutrl)   

        input_ids_w1_c = self._tokenizer.encode(contxt_w1, return_tensors="pt")
        input_list_w1_c = input_ids_w1_c.cpu().detach().numpy().tolist()[0]
        input_lists_w1_c = [input_list_w1_c]
        input_ids_w1_c = torch.LongTensor(input_lists_w1_c) 

        input_ids_w2_c = self._tokenizer.encode(contxt_w2, return_tensors="pt")
        input_list_w2_c = input_ids_w2_c.cpu().detach().numpy().tolist()[0]
        input_lists_w2_c = [input_list_w2_c]
        input_ids_w2_c = torch.LongTensor(input_lists_w2_c)   

        input_ids_w3_c = self._tokenizer.encode(contxt_w3, return_tensors="pt")
        input_list_w3_c = input_ids_w3_c.cpu().detach().numpy().tolist()[0]
        input_lists_w3_c = [input_list_w3_c]
        input_ids_w3_c = torch.LongTensor(input_lists_w3_c)   

        past_w1, past_w2, past_w3, past_actual, past_cntr_2, past_neutrl, past_ac_sdb, past_cn_sdb, use_cache = None, None, None, None, None, None, None, None, True
        sent = prompt_w1

        reqd_indices_w1, reqd_indices_w2 = self.find_reqd_indices(torch.flip(input_ids_w1[0], dims=[-1]), torch.flip(input_ids_w2[0], dims=[-1]))
        reqd_in_cntx_w1, reqd_in_cntx_w2 = self.find_reqd_indices(torch.flip(input_ids_w1_c[0], dims=[-1]), torch.flip(input_ids_w2_c[0], dims=[-1]))
        reqd_indices_w3 = reqd_indices_w2
        reqd_in_cntx_w3 = reqd_in_cntx_w2

        if self.ct != 'n':
            input_ids_w1 = torch.cat([input_ids_w1_c, input_ids_w1], dim=-1)
            input_ids_w2 = torch.cat([input_ids_w2_c, input_ids_w2], dim=-1)
            input_ids_w3 = torch.cat([input_ids_w3_c, input_ids_w3], dim=-1)

        ignore_list = []
        for word in self.words_ignore:
            ignore_list.append(' ' + word)
        ignore_ids_list = []
        for word in ignore_list:
            w_ids = self._tokenizer.encode(word)
            for w in w_ids:
                ignore_ids_list.append(w)

        reqd_indices_w1 = [x + len(input_ids_w1_c[0]) for x in reqd_indices_w1]
        reqd_indices_w2 = [x + len(input_ids_w2_c[0]) for x in reqd_indices_w2]
        reqd_indices_w3 = [x + len(input_ids_w3_c[0]) for x in reqd_indices_w3]

        reqd_indices_w1 = reqd_in_cntx_w1 + reqd_indices_w1
        reqd_indices_w2 = reqd_in_cntx_w2 + reqd_indices_w2
        reqd_indices_w3 = reqd_in_cntx_w3 + reqd_indices_w3

        reqd_indices_w1.append(len(input_ids_w1[0])-1)
        reqd_indices_w2.append(len(input_ids_w2[0])-1)
        reqd_indices_w3.append(len(input_ids_w3[0])-1)

        len_w1 = len(input_ids_w1[0])
        len_w2 = len(input_ids_w2[0])
        len_w3 = len(input_ids_w3[0])

        indices_skipped_w1 = []
        indices_skipped_w2 = []
        indices_skipped_w3 = []

        for i in range(len(input_ids_w1[0])):
            if i not in reqd_indices_w1:
                indices_skipped_w1.append(i)

        for i in range(len(input_ids_w2[0])):
            if i not in reqd_indices_w2:
                indices_skipped_w2.append(i)

        for i in range(len(input_ids_w3[0])):
            if i not in reqd_indices_w3:
                indices_skipped_w3.append(i)

        jsp = []

        for i in range(sent_len):
            probs_w1, past_w1, outputs_w1, ro_w1 = self.generate_probabilities (input_ids_w1, past_w1, attention_mask_w1, use_cache)
            probs_w2, past_w2, outputs_w2, ro_w2 = self.generate_probabilities (input_ids_w2, past_w2, attention_mask_w2, use_cache)
            probs_w3, past_w3, outputs_w3, ro_w3 = self.generate_probabilities (input_ids_w3, past_w3, attention_mask_w3, use_cache)

            outputs_w1 = outputs_w1.to(device)

            outputs_w1_trimmed = torch.index_select(outputs_w1.to(device), 1, torch.Tensor(reqd_indices_w1).int().to(device))
            outputs_w2_trimmed = torch.index_select(outputs_w2.to(device), 1, torch.Tensor(reqd_indices_w2).int().to(device))
            outputs_w3_trimmed = outputs_w2_trimmed
            ids_w1_trimmed = torch.index_select(input_ids_w1.to(device), -1, torch.Tensor(reqd_indices_w1).int().to(device)).to(device)
            scores, avg, scres2 = self.calculate_scores_1 (outputs_w1_trimmed, outputs_w2_trimmed, outputs_w3_trimmed, self.rat)

            for idx in range(1,len(ids_w1_trimmed[0])):
                jsp.append(scores[0, idx-1, ids_w1_trimmed[0][idx]].item())

            da = None
            probs_new, da = self.calculate_new_pdf(outputs_w1_trimmed, scores, scres2, avg, self.thres, self.sf, self.lmbd, ignore_ids_list)

            outputs_w1[0][torch.Tensor(reqd_indices_w1).long()] = probs_new.to(device)   #.to(dtype=torch.bfloat16)
            outputs_actual = self.alpha*outputs_w1[:,-l:,:] + (1-self.alpha)*torch.nn.functional.softmax(ro_w1[:, -l:, :]/self.temperature, dim=-1)  #outputs_w1_copy
            logit_id = torch.multinomial(outputs_actual[0][-1], num_samples=1).to(device)[0]
            word = self._tokenizer.decode(logit_id)
            sent += word

            input_ids_w1 = torch.cat([input_ids_w1.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            attention_mask_w1 = torch.cat([attention_mask_w1, attention_mask_w1.new_ones((attention_mask_w1.shape[0], 1))], dim=-1)
            input_ids_w2 = torch.cat([input_ids_w2.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            attention_mask_w2 = torch.cat([attention_mask_w2, attention_mask_w2.new_ones((attention_mask_w2.shape[0], 1))], dim=-1)
            input_ids_w3 = torch.cat([input_ids_w3.to(device), logit_id.to(device).unsqueeze(-1).unsqueeze(-1)], dim=-1)
            attention_mask_w3 = torch.cat([attention_mask_w3, attention_mask_w2.new_ones((attention_mask_w3.shape[0], 1))], dim=-1)

            reqd_indices_w1.append(i+len_w1)
            reqd_indices_w2.append(i+len_w2)
            reqd_indices_w3.append(i+len_w3)

        return outputs_actual, np.mean(jsp), sent, da