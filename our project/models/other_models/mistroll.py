from typing import List, Union
import random
import os

import torch.cuda

from .base_model import ShopBenchBaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import spacy

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class Mistroll(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
       # model_path = 'lmsys/vicuna-7b-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained('./models/mistroll/', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained('./models/mistroll/',  device_map='auto',
                                                          torch_dtype='auto', trust_remote_code=True, do_sample=True)
        #self.model.to('cuda')
        #self.system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
        #self.system_prompt = "Listen to the given task carefully and pay special attention on how the answers should be given.\n\n"
        self.system_prompt = "Listen to the given task carefully and pay special attention on how the answers should be given. "


    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        if is_multiple_choice:
            prompt = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n\n" + prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=1, temperature=0.8)


        else:
            prompt = self.system_prompt +"Do not repeat the task and also do not explain your though process.\n\n" + prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs=inputs.input_ids,
                                               max_new_tokens=100,
                                               temperature=0.9)

        generate_ids= [
            output_ids[len(inputs.input_ids):] for inputs.input_ids, output_ids in zip(inputs.input_ids, generate_ids)
        ]
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #generation = result[len(prompt):]
        return result

