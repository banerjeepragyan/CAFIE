from transformers import PretrainedConfig

class SelfDebiasingAutoConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init(**kwargs)
