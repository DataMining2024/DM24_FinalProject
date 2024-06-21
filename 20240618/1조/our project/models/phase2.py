from typing import List, Union
from typing import Any, Dict, List
import random
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import vllm

from .base_model import ShopBenchBaseModel

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))

# Set a consistent seed for reproducibility
#AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 16 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
class Dareties(ShopBenchBaseModel):
    def __init__(self):
        random.seed(AICROWD_RUN_SEED)
        self.initialize_models()

    def initialize_models(self):

        self.model_name = "models/daretires"
        #self.tokenizer = AutoTokenizer.from_pretrained('./models/daretires/', trust_remote_code=True)
        #self.model = AutoModelForCausalLM.from_pretrained('./models/daretires/', device_map='auto',
        #                                                  torch_dtype=torch.float16, trust_remote_code=True,
        #                                                  do_sample=True)
        self.system_prompt = "Please listen to the given task carefully and pay special attention on how the answers should be give.\n"
        #self.system_prompt = ""

        self.llm = vllm.LLM(
            self.model_name,
            worker_use_ray=True,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()


    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_predict` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_predict calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice: bool) -> List[str]:
        """
        Generates a batch of prediction based on associated prompts and task_type

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Parameters:
            - batch (Dict[str, Any]): A dictionary containing a batch of input prompts with the following keys
                - prompt (List[str]): a list of input prompts for the model.

            - is_multiple_choice bool: A boolean flag indicating if all the items in this batch belong to multiple choice tasks.

        Returns:
            str: A list of predictions for each of the prompts received in the batch.
                    Each prediction is
                           a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """
        prompts = batch["prompt"]

        # format prompts using the chat template
        formatted_prompts = self.format_prommpts(prompts, is_multiple_choice)
        # set max new tokens to be generated
        max_new_tokens = 100

        if is_multiple_choice:
            max_new_tokens = 1  # For MCQ tasks, we only need to generate 1 token

        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=1,  # randomness of the sampling
                seed=AICROWD_RUN_SEED,  # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm=False
        )
        # Aggregate answers into List[str]
        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)

        if is_multiple_choice:
            print("MCQ: ", batch_response)

        return batch_response

    def format_prommpts(self, prompts, is_multiple_choice: bool):
        """
        Formats prompts using the chat_template of the model.

        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.

        """
        #system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
        formatted_prompts = []
        for prompt in prompts:

            if is_multiple_choice:

                if ("Which of the following product" and "attribute") in prompt or (
                        "Select the category") in prompt:  # task 2
                    prompt_edit = (
                                      "You will be given a specific term. This term is an attribute of ONE of the following product categories. You have to choose to which term this attribute may fit.\n\n"
                                      "Follow this example: \n 2.	Which of the following product categories may have the attribute trigger.\n 0. football1. selfie stick\n 2. broom\n3. calend\n\nOutput: 1\n\n") + prompt


                elif ("Select the category" and "attribute") in prompt:  # task 2
                    prompt_edit = (
                                      "You will be given a specific attribute. You have to choose, which one of the following product categories could have this attribute.\n") + prompt

                elif ("e-commerce website" and "What type of") in prompt or (
                        "e-commerce website" and " Is the") in prompt:  # task 5 // maybe have to change
                    prompt_edit = ("You will be given a product. Answer the given question based on the information "
                                   "you have. \n") + prompt

                elif ("e-commerce website" and "total" in prompt) or (
                        "e-commerce website" and "How many") in prompt:  # task 8 // might has to change    - mehr drauf eingehen wie zu rechnen/worauf aufpassen
                    prompt_edit = (("You will be given a product. Answer the given question based on the information "
                                    "you have. You have to calculate the answer based on the product name. \n"
                                    "Follow this example: \n\n The product 'One use Cups, White or with Logo, 300 Count, Pack of 3' appears on e-commerce website. What is the total count of disposable napkins in this package?\n"
                                    "0. 300 count\n1. 660 count\n2. 900 count\n3. 1000 count\nOutput: 2\n") + prompt)

                elif "PersonX" in prompt:  # task 9
                    prompt_edit = (
                                      "You have to make a decision for the given event. The decision should be the most logical cause one would expect. There is always a correct answer, which is never -1.\n") + prompt

                elif "Which of the following product categories best complement the product type" in prompt or "which of the following product categories best complement the given product type?" in prompt:  # task 10
                    prompt_edit = (
                                      "You have to choose a category for the following product. The category chosen should be the most complementing for the given product.\n") + prompt

                elif "Which of the following statements" in prompt:  # task 11
                    prompt_edit = ("You will be given two queries. You have to decide, in which way they are "
                                   "related. Take special attention what the general categories of the given queries "
                                   "are and how they relate towards each other. \n") + prompt

                elif "Evaluate the following product review on a scale of 1 to 5" in prompt:  # task 15
                    prompt_edit = (
                                      "You will be given a product and a review given for it. Rate the review on a scale from 1 to 5, with 1 being a very negative review and 5 a very positive review.\n") + prompt

                elif ("following product" and "following keyword" and "most suitable") in prompt or (
                        "following sets of phrases" and "summarize" and "following product") in prompt:  # task 16
                    prompt_edit = (
                                      "You will be given a product title and description in a foreign language. Choose which of the following categories describes the product the best.\n") + prompt

                elif (
                        "description" and "exists" and " describe" and "same product" and "different language") in prompt:  # task 18
                    prompt_edit = (
                                      "You will be given a product with it's description. You have to choose, which of the following products may describe the same product, but in a different language.\n") + prompt

                elif (
                        "user" and "found" and "description" and "another shopping website in a different language") in prompt:  # task 18
                    prompt_edit = (
                                      "You will  be given a product description. You have to choose, which of the following descriptions in another language is the most fitting for the given product description. There is always a correct answer, which is never -1.\n") + prompt

                else:
                    prompt_edit = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n" + prompt


            else:

                if (("xplain " and "category name") in prompt or ("xplain " and "product type") in prompt or (
                        "ell " and "product category") in prompt):  # task 1 sagen, soll wie definition sein. also mit wort anfangen oder a ... is a...
                    prompt_edit = (
                                      "You have to explain the given product category or type. Your answer should be given "
                                      "in one sentence of a length up to 30 words. It should also be given like a definition, starting with the word or with 'A ... is'.\n") + prompt  # 20 words? 30 words?

                elif "sentiment" and "potential review" in prompt:  # task 3
                    prompt_edit = ("Listen to the given task carefully and pay attention how the formatting will "
                                   "be. Take special attention on which kind of product it is and which of the "
                                   "reviews fits the aspect that should be mentioned. \n") + prompt

                elif "extract phrases" in prompt:  # task 4  sagen, nur ein einzelnes wort von der auswahl, keine weiteren mit komma getrennt oder auch keine wortgruppen
                    prompt_edit = ("Listen to the given task in the next paragraph carefully.  "
                                   "Write ONLY the SINGLE most fitting word from the query."
                                   "This word should be the best fitting given entity type in inverted commas. There should be no other words selected, also not divided by commas. Do not give any reasoning. The chosen word should be written in small letters only. You should NOT use character other than letters.\n"
                                   "Follow these two examples: \n\nYou are a helpful online shop assistant and a linguist. A customer on an online shopping platform has made the following query. Please extract phrases from the query that correspond to the entity type 'vacation'. Please directly output the entity without repeating the entity type. If there are multiple such entities, separate them with comma. Do not give explanations. \n"
                                   "Query: medication ticket 5m wrap\n Output: ticket\n\n"
                                   "You are a helpful online shop assistant and a linguist. A customer on an online shopping platform has made the following query. Please extract phrases from the query that correspond to the entity type 'product type'. Please directly output the entity without repeating the entity type. If there are multiple such entities, separate them with comma. Do not give explanations. \n"
                                   "Query: tablette apple\nOutput: tablette") + prompt

                elif "extract the keyphrase" in prompt:  # task 6  maybe: soll nicht laenger als 5 woerter sein? und sagen, soll nur aus selben satz sein, plus keine explaination
                    prompt_edit = (
                                      "You will be given a review and an aspect. You should generate excatly one short keyphrase based on the given aspect. The keyphrase should only contain the exact words of the review and not be a full sentence. The keyphrase should also be from the same sentence. You should not explain the answer or generate other irrelevant text. \n") + prompt

                elif "You are given a user review given to a(n) " and "Please choose three aspects from the list that are covered by the review" in prompt:  # task 7
                    prompt_edit = (
                                      "You will be given a list and a specific review. You should EXACTLY choose 3 aspects as numbers from the given list that are covered in the review and nothing else.\n") + prompt

                elif " rank " and "product" and "relevance" in prompt:  # task 12
                    prompt_edit = ("You have to rank the given products based on the query that is about to be "
                                   "mentioned. The way you should answer is described at the end of the "
                                   "task. You ALWAYS have to give all 5 numbers.\n") + prompt

                elif ("user" and "sequence" and "queries" and "keyword") in prompt:  # task 13
                    prompt_edit = (
                                      "Follow the given scenario of you being a user. You just clicked through an online store and made purchases. You are given a query sequence, on which you should guess THREE next most likely queries you would click on, based on your previous interests.\n") + prompt

                elif ("user" in prompt and "may also buy" in prompt and "product" in prompt):  # task 13
                    prompt_edit = (
                                      "You will be given a product, which just has been bought. Now choose from the following product list EXACTLY 3 products, which are the closest to the given product.\n") + prompt

                elif ("user" in prompt and "may also purchase" in prompt and "product" in prompt):  # task 14
                    prompt_edit = (
                                      "You will be given a product, which just has been bought. Now choose from the following product list EXACTLY 3 products, which are the closest to the given product.\n") + prompt

                elif "adequate title" in prompt:  # task 17
                    prompt_edit = (
                                      "You will be given a product title in inverted commas. You should write the title in the given based on the given context in the language that is mentioned. Do not explain your decisions.\n ") + prompt
                    # You should translate the title into the given language based on how it would appear on an online shopping website.

                elif ("ranslate " and "title" and "into") in prompt:  # task 17
                    prompt_edit = (
                                      "You will be given a product title in inverted commas. You have to translate it based on the "
                                      "language given in the description. You should not explain the answer. \n") + prompt

                else:
                    prompt_edit = self.system_prompt + prompt

            formatted_prompts.append(prompt_edit)

        return formatted_prompts



    def predict(self, prompt: str, is_multiple_choice: bool) -> str:

        if is_multiple_choice:

            if ("Which of the following product" and "attribute") in prompt or (
            "Select the category") in prompt:  # task 2
                prompt_edit = ("You will be given a specific term. This term is an attribute of ONE of the following product categories. You have to choose to which term this attribute may fit.\n\n"
                               ) + prompt


            elif( "Select the category" and "attribute") in prompt:  # task 2
                prompt_edit = ("You will be given a specific attribute. You have to choose, which one of the following product categories could have this attribute.\n") + prompt

            elif ("e-commerce website" and "What type of") in prompt or (
                    "e-commerce website" and " Is the") in prompt:  # task 5 // maybe have to change
                prompt_edit = ("You will be given a product. Answer the given question based on the information "
                               "you have. \n") + prompt

            elif ("e-commerce website" and "total" in prompt) or (
                    "e-commerce website" and "How many") in prompt:  # task 8 // might has to change    - mehr drauf eingehen wie zu rechnen/worauf aufpassen
                prompt_edit = (("You will be given a product. Answer the given question based on the information "
                                "you have. You have to calculate the answer based on the product name. \n") + prompt)

            elif "PersonX" in prompt:  # task 9
                prompt_edit = (
                                  "You have to make a decision for the given event. The decision should be the most logical cause one would expect. There is always a correct answer, which is never -1.\n") + prompt

            elif "Which of the following product categories best complement the product type" in prompt or "which of the following product categories best complement the given product type?" in prompt:  # task 10
                prompt_edit = (
                                  "You have to choose a category for the following product. The category chosen should be the most complementing for the given product.\n") + prompt

            elif "Which of the following statements" in prompt:  # task 11
                prompt_edit = ("You will be given two queries. You have to decide, in which way they are "
                               "related. Take special attention what the general categories of the given queries "
                               "are and how they relate towards each other. \n") + prompt

            elif "Evaluate the following product review on a scale of 1 to 5" in prompt:  # task 15
                prompt_edit = (
                                  "You will be given a product and a review given for it. Rate the review on a scale from 1 to 5, with 1 being a very negative review and 5 a very positive review.\n") + prompt

            elif ("following product" and "following keyword" and "most suitable") in prompt or (
                    "following sets of phrases" and "summarize" and "following product") in prompt:  # task 16
                prompt_edit = (
                                  "You will be given a product title and description in a foreign language. Choose which of the following categories describes the product the best.\n") + prompt

            elif (
                    "description" and "exists" and " describe" and "same product" and "different language") in prompt:  # task 18
                prompt_edit = (
                                  "You will be given a product with it's description. You have to choose, which of the following products may describe the same product, but in a different language.\n") + prompt

            elif (
                    "user" and "found" and "description" and "another shopping website in a different language") in prompt:  # task 18
                prompt_edit = (
                                  "You will  be given a product description. You have to choose, which of the following descriptions in another language is the most fitting for the given product description. There is always a correct answer, which is never -1.\n") + prompt

            else:
                prompt_edit = self.system_prompt + "The given question is a multiple choice question. Pay special care on how you should answer.\n" + prompt

            prompt_edit = self.system_prompt + prompt_edit
            inputs = self.tokenizer(prompt_edit, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=1, temperature=1)

        else:

            if (("xplain " and "category name") in prompt or ("xplain " and "product type") in prompt or (
                    "ell " and "product category") in prompt):  # task 1 sagen, soll wie definition sein. also mit wort anfangen oder a ... is a...
                prompt_edit = ("You have to explain the given product category or type. Your answer should be given "
                               "in one sentence of a length up to 30 words. It should also be given like a definition, starting with the word or with 'A ... is'.\n") + prompt  # 20 words? 30 words?

            elif "sentiment" and "potential review" in prompt:  # task 3
                prompt_edit = ("Listen to the given task carefully and pay attention how the formatting will "
                               "be. Take special attention on which kind of product it is and which of the "
                               "reviews fits the aspect that should be mentioned. \n") + prompt

            elif "extract phrases" in prompt:  # task 4  sagen, nur ein einzelnes wort von der auswahl, keine weiteren mit komma getrennt oder auch keine wortgruppen
                prompt_edit = ("Listen to the given task in the next paragraph carefully.  "
                               "Write ONLY the SINGLE most fitting word from the query."
                               "This word should be the best fitting given entity type in inverted commas. There should be no other words selected, also not divided by commas. Do not give any reasoning. The chosen word should be written in small letters only. You should NOT use character other than letters.\n"
                              ) + prompt

            elif "extract the keyphrase" in prompt:  # task 6  maybe: soll nicht laenger als 5 woerter sein? und sagen, soll nur aus selben satz sein, plus keine explaination
                prompt_edit = (
                                  "You will be given a review and an aspect. You should generate excatly one short keyphrase based on the given aspect. The keyphrase should only contain the exact words of the review and not be a full sentence. The keyphrase should also be from the same sentence. You should not explain the answer or generate other irrelevant text. \n") + prompt

            elif "You are given a user review given to a(n) " and "Please choose three aspects from the list that are covered by the review" in prompt:  # task 7
                prompt_edit = (
                                  "You will be given a list and a specific review. You should EXACTLY choose 3 aspects as numbers from the given list that are covered in the review and nothing else.\n") + prompt

            elif " rank " and "product" and "relevance" in prompt:  # task 12
                prompt_edit = ("You have to rank the given products based on the query that is about to be "
                               "mentioned. The way you should answer is described at the end of the "
                               "task. You ALWAYS have to give all 5 numbers.\n") + prompt

            elif ("user" and "sequence" and "queries" and "keyword") in prompt:  # task 13
                prompt_edit = (
                                  "Follow the given scenario of you being a user. You just clicked through an online store and made purchases. You are given a query sequence, on which you should guess THREE next most likely queries you would click on, based on your previous interests.\n") + prompt

            elif ("user" in prompt and "may also buy" in prompt and "product" in prompt): # task 13
                prompt_edit = (
                                  "You will be given a product, which just has been bought. Now choose from the following product list EXACTLY 3 products, which are the closest to the given product.\n") + prompt

            elif ("user" in prompt and "may also purchase" in prompt and "product" in prompt):  # task 14
                prompt_edit = (
                                  "You will be given a product, which just has been bought. Now choose from the following product list EXACTLY 3 products, which are the closest to the given product.\n") + prompt

            elif "adequate title" in prompt:  # task 17
                prompt_edit = (
                                  "You will be given a product title in inverted commas. You should write the title in the given based on the given context in the language that is mentioned. Do not explain your decisions.\n ") + prompt
                # You should translate the title into the given language based on how it would appear on an online shopping website.

            elif ("ranslate " and "title" and "into") in prompt:  # task 17
                prompt_edit = (
                                  "You will be given a product title in inverted commas. You have to translate it based on the "
                                  "language given in the description. You should not explain the answer. \n") + prompt

            else:
                prompt_edit = self.system_prompt + prompt

            prompt_edit = self.system_prompt + prompt_edit
            #print("prompt:_----------------------------_" + prompt_edit)
            inputs = self.tokenizer(prompt_edit, return_tensors='pt')
            inputs.input_ids = inputs.input_ids.cuda()
            generate_ids = self.model.generate(inputs=inputs.input_ids, max_new_tokens=100,
                                               temperature=1)  # temperature probieren zu aendern?

        result = \
            self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation = result[len(prompt_edit):]

        return generation

    # {"input_field":"A product entitled 'JETech Case for iPad (9.7-Inch, 2018\/2017 Model, 6th\/5th Generation), Smart Cover Auto Wake\/Sleep (Light Purple)' exists on an online shopping website. Generate an adequate title for the product when it appears on a(n) Japanese online shopping website.\nOutput: ","output_field":"JEDirect iPad 9.7\u30a4\u30f3\u30c1 (2018\/2017 \u7b2c6\/5\u4e16\u4ee3\u7528) \u30b1\u30fc\u30b9 PU\u30ec\u30b6\u30fc \u4e09\u3064\u6298\u30b9\u30bf\u30f3\u30c9 \u30aa\u30fc\u30c8\u30b9\u30ea\u30fc\u30d7\u6a5f\u80fd (\u30e9\u30a4\u30c8\u30d1\u30fc\u30d7\u30eb)","task_name":"task17","task_type":"generation","metric":"jp-bleu","is_multiple_choice":false,"track":"amazon-kdd-cup-24-multi-lingual-abilities"}
