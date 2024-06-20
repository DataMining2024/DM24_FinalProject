from typing import List, Union
import random
import os

from .base_model import ShopBenchBaseModel
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModel
from transformers import BartForSequenceClassification, BartTokenizer
from torch.nn import functional as f




# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))



class MyModel(ShopBenchBaseModel):
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """
    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)

        #need: sentence + labels


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

        """
        test?
        
        #classifier = pipeline('albert', )
        first_classifier = pipeline(model="FacebookAI/roberta-large-mnli", task ='text-classification') # sentiment-analysis ?
        #model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
        """
        """ tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        model = AutoModel.from_pretrained('facebook/bart-large-mnli')

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        logits = model(input_ids)[0]

        entail_contradiction_logits = logits[:, [0,2]]
        print(entail_contradiction_logits)
        """


        possible_responses = [1, 2, 3, 4]

        if is_multiple_choice:
            # Randomly select one of the possible responses for multiple choice tasks
            print("random")
            return str(random.choice(possible_responses))
        else:
            #likely_response = first_classifier(prompt, possible_responses)
           ### tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
            ###model = AutoModel.from_pretrained('deepset/sentence_bert')
            ###first_classifier = pipeline(model='facebook/bart-large-mnli', task="zero-shot-classification")
            #sentiment-analysis, text-classification, zero-shot-classification

            ###results = first_classifier(prompt,possible_responses)
            ###categories = results["labels"]
            ###scores = results["scores"]

            tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
            model = AutoModel.from_pretrained('facebook/bart-large-mnli')

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            logits = model(input_ids)[0]

            entail_contradiction_logits = logits[:, [0, 2]]

            #s = 0
            #c = ""
            #print(results)
            #for category, probability in zip(categories, scores):
            #    print(f"{category}: {probability:.4f}")
            #    if s < probability:
            #        s = probability
            #        c = category
            print(entail_contradiction_logits)
            return str(entail_contradiction_logits)




"""
#development data set: json, 'input_field', This field contains the instructions and the question that should be answered by the model.
#       output_field': This field contains the ground truth answer to the question.
#       'task_type': This field contains the type of the task (Details in the next Section, "Tasks")
        'metric': This field contains the metric used to evaluate the question (Details in Section "Evaluation Metrics"). 
    In test set only:
        'input_field', which is the same as above. 
        'is_multiple_choice': This field contains a 'True' or 'False' that indicates whether the question is a multiple choice or not. The detailed 'task_type' will not be given to participants.   
    
    


"""