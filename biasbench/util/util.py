def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "DropoutAutoModelForCausalLM",
        "SelfDebiasGPT2LMHeadModel",
        "AutoModelForCausalLM",
        "SelfDebiasAutoModelForCausalLM",
        "SentenceDebiasGPT2Model",
        "SentenceDebiasAutoModelForCausalLM"
    ]


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return model in [
        "SelfDebiasGPT2LMHeadModel",
        "SelfDebiasBertForMaskedLM",
        "SelfDebiasAlbertForMaskedLM",
        "SelfDebiasRobertaForMaskedLM",
        "SelfDebiasAutoModelForCausalLM"
    ]
