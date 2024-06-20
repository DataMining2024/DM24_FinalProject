from typing import List, Union
import random
import os
from datasets import load_dataset
import pandas as pd
import json


dataset = load_dataset("NingLab/ECInstruct")['train'].shuffle(seed=42).select(range(10000))
df_train = pd.DataFrame(dataset)

formatted_data = []

for index, row in df_train.iterrows():

    #if row["split"] == 'train':
        if row["options"] != 'None':
            formatted_entry = {
                'input_field': row['instruction']  + "\n" + str(row["input"]) + "\n Options:" +  str(row["options"]),
                'output': row["output"],
                'type': row['split']
            }
        else: formatted_entry = {
                'input_field': row['instruction']  + "\n" + str(row["input"]),
                'output': row["output"],
                'type': row['split']
            }
        formatted_data.append(formatted_entry)

with open('ECinstruct_train.json', 'w') as outfile:
    json.dump(formatted_data, outfile, indent=4)





