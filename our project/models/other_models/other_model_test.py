from typing import List, Union
import random
import os

import torch.cuda

from .base_model import ShopBenchBaseModel
from transformers import pipeline, AutoTokenizer, logging, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import spacy


# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))

access_token = "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"

use_triton = False

class Vicuna2_test(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        model_path = 'lmsys/vicuna-13b-v1.3'
        #model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        
        
        """
           model_name_or_path = "TheBloke/vicuna-13b-v1.3.0-GPTQ"
        #model_basename = "vicuna-13b-v1.3.0-GPTQ-4bit-128g.no-act.order"
        model_basename = "gptq-8bit-128g-actorder_False"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)
        """

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            
        

        #pipelines = pipeline(
        #    "text-generation",
        #    model=model_id,
        #    model_kwargs={"torch_dtype": torch.bfloat16},
        #    device_map="auto",
        #)
        #self.tokenizer = GPT2Tokenizer.from_pretrained(model_id, token=access_token)
        #self.model = GPT2LMHeadModel.from_pretrained(model_id, token=access_token)


        #self.tokenizer = AutoTokenizer.from_pretrained('./models/vicuna-7b-v1.5/', trust_remote_code=True)
        #self.model = AutoGPTQForCausalLM.from_pretrained('./models/vicuna-7b-v1.5/', quantize_config=None, device_map='auto',
        #                                                  torch_dtype='auto', trust_remote_code=True, do_sample=True)
        
        #self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        #self.model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=None, device_map='auto',
        #                                                  torch_dtype='auto', trust_remote_code=True, do_sample=True)
        #self.model.to('cuda')
        #self.system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
        #self.system_prompt = "Listen to the given task carefully and pay special attention on how the answers should be given.\n\n"
        self.system_prompt = "Listen to the given task carefully and pay special attention on how the answers should be given.\n\n"


    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        if is_multiple_choice:
            prompt = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n\n" + prompt
        else:
            prompt = self.system_prompt + prompt
            
        output = pipe(prompt, **generation_args)
        return output[0]['generated_text']

        """
        if is_multiple_choice:
            prompt = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n\n" + prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            #generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=1, temperature=0)
            output = self.model.generate(inputs=inputs.input_ids, temperature=0, max_new_tokens=1)


        else:
            prompt = self.system_prompt + prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            #generate_ids = self.model.generate(inputs = inputs.input_ids,
            #                                   max_new_tokens=100,
            #                                   temperature=0)
            output = self.model.generate(inputs=inputs.input_ids, temperature=0.7, max_new_tokens=512)
        
        """
    


        
        #result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #terminators = [
        #pipeline.tokenizer.eos_token_id,
        #pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        #]
        #outputs = pipeline(
        #    prompt,
        #    max_new_tokens=256,
        #    eos_token_id=terminators,
        #    do_sample=True,
        #    temperature=0.6,
        #    top_p=0.9,
        #)

        #pipe = pipeline(
        #    "text-generation",
        #    model=model,
        #    tokenizer=tokenizer,
        #    max_new_tokens=512,
        #    temperature=0.7,
        #    top_p=0.95,
        #    repetition_penalty=1.15
        #)

        generation = self.tokenizer.decode(output[0])

        #generation = self.tokenizer.decode(generate_ids[0]["generated_text"][len(prompt):])
        #generation = self.tokenizer.decode(generate_ids[0],skip_special_tokens=True)
        #generation = result[len(prompt):]
        return generation

