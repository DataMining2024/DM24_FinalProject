from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam



dataset = load_dataset("ECinstruct_train.json")
dataset = dataset["train"]  # Just take the training split for now



tokenizer = AutoTokenizer.from_pretrained('./mistroll/', trust_remote_code=True)
tokenized_data = tokenizer(dataset["input_field"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1

# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained('./mistroll/', trust_remote_code=True)
# Lower learning rates are often better for fine-tuning transformers
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tokenized_data, labels)
