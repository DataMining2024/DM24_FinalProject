from typing import List, Union
import random
import os

import torch.cuda
device = torch.device("cpu")
from .base_model import ShopBenchBaseModel
from transformers import pipeline, AutoTokenizer, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import spacy


print(torch.cuda.is_available())
# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))


class MyModel2(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)

    def predict(self, prompt: str, is_multiple_choice: bool) -> str:
        """
        Generates a prediction based on the input prompt and task type.

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Args:
            prompt (str): The input prompt for the model.
            is_multiple_choice (bool): Indicates whether the task is a multiple choice question.

        Returns:
            str: The prediction as a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        possible_responses = [1, 2, 3, 4]

        model_name_or_path = ('TheBloke/vicuna-7B-v1.3-GPTQ')
       # model_basename = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
        model_basename = "model"
        use_triton = False



        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast= True)
        model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                                   model_basename = model_basename,
                                                   use_safetensors=True,
                                                   trust_remote_code=True,
                                                   device='cuda:0',
                                                   use_triton=use_triton,
                                                   quantize_config=None,
                                                   )

        #prompt_template = f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        #
        #USER: {prompt}
        #ASSISTANT:
        #'''
        system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
        prompt_template  = system_prompt + prompt
        print("\n\n*** Generate:")

        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids,temperature=0.7, max_new_tokens=512)
        print(tokenizer.decode(output[0]))

        logging.set_verbosity(logging.CRITICAL)

        print("*** Pipeline:")
        pipe = pipeline("text-generation",
                        model= model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        repetition_penalty=1.15)
        print(pipe(prompt_template)[0]['generated_text'])




        #generator = pipeline('text-generation')

        #generated_text = generator(prompt, do_sample="true", temperature= 0.7, max_length = 500)
        #print("generated_text", generated_text[0]["generated_text"])
        if is_multiple_choice:
            # Randomly select one of the possible responses for multiple choice tasks
            #return str(random.choice(possible_responses))
            return str(pipe(prompt_template)[0]['generated_text'])
        else:
            # For other tasks, shuffle the possible responses and return as a string
            random.shuffle(possible_responses)
            return str(pipe(prompt_template)[0]['generated_text'])
            # Note: As this is dummy model, we are returning random responses for non-multiple choice tasks.
            # For generation tasks, this should ideally return an unconstrained string.

