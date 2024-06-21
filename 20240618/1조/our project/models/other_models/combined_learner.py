from datasets import load_dataset, Dataset
import json
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

pd.read_json("ECinstruct_train.json", lines=True)
# Load your dataset
with open('ECinstruct_train.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)



model_name = "mistroll"
tokenizer = AutoTokenizer.from_pretrained('./mistroll/', trust_remote_code=True)

def tokenize_function(examples):
    return tokenizer(examples["input_field"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained('./mistroll/', trust_remote_code=True, num_labels=len(data[0]['input_field']))


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16 = True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

#train model
trainer.train()

#eval model
results = trainer.evaluate()
print(results)

#save fine-tuned model
model.save_pretrained("./mymodel/fine-tuned-model")
tokenizer.save_pretrained("./mymodel/fine-tuned-model")