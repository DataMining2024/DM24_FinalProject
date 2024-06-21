from typing import List, Union
import random
import os

import torch.cuda

from .base_model import ShopBenchBaseModel
from transformers import pipeline, AutoTokenizer, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import spacy

print(torch.version.cuda)
print(torch.cuda.is_available())
# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class Vicunatest(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        model_path = 'lmsys/vicuna-7b-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained('./models/vicuna-7b-v1.5/', trust_remote_code=True)
        self.model = AutoGPTQForCausalLM.from_pretrained('./models/vicuna-7b-v1.5/', quantize_config=None, device_map='auto',
                                                          torch_dtype='auto', trust_remote_code=True, do_sample=True)
        self.model.to('cuda')
        #self.system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
        #self.system_prompt = "Listen to the given task carefully and pay special attention on how the answers should be given.\n\n"
        self.system_prompt = "Please analyse, what the task of the given prompt is. Be as detailed as possible, whether it is multiple choice or not, a selection, description, selection or so on. Give the answer out in one sentence,\n\n"

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:

        prompt_first = self.system_prompt + prompt
        inputs_first = self.tokenizer(prompt_first, return_tensors='pt')
        inputs_first.input_ids = inputs_first.input_ids.cuda()
        generate_ids_first = self.model.generate(inputs=inputs_first.input_ids, max_new_tokens=1, temperature=0)\

        result_first =  self.tokenizer.batch_decode(generate_ids_first, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation_first = result_first[len(prompt):]
       # new_prompt = generate_ids_first + "\n\n" + prompt

        if is_multiple_choice:
            #prompt = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n\n" + prompt
            prompt = generation_first + "\n\n" + prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=1, temperature=0)


        else:
            prompt = generation_first + "\n\n" + prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs = inputs.input_ids,
                                               max_new_tokens=100,
                                               temperature=0)
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation = result[len(prompt):]
        return generation

