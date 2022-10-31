import numpy as np
import torch.nn
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from tqdm import trange
from datasets import ClassLabel, load_dataset
import json
from datasets import Dataset, load_dataset
import logging
from tqdm import trange
from pynvml import *

logging.basicConfig(level=logging.INFO)
torch.cuda.empty_cache()


def load_examples(dataset_file):
    with open(dataset_file, "r") as f:
        data = json.load(f)
    examples = []
    for item in data:
        examples.append(dict(
            text=item["sent"],
            entity_spans=[(item["start"], item["end"])],
            label=item["labels"]
        ))
    return examples


test_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/new_convertLUKE/OpenEntity/test.json")

logging.info("Data Memory before Loading Models")
#print_gpu_utilization()
logging.info("############### LOAD MODEL ###################")
#my_config = LukeConfig.from_json_file("./ET_for_FIGER_model_v2/config.json")
#print(my_config)
# model LongLuke NoGlobal - OpenEntity
#model = LukeForEntityClassification.from_pretrained("/mnt/shared/home/jose.luis.malaquias.ext/new_convertLUKE/LongLukeOpenEntity/Vanilla_long_3Agosto")
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
#model.config = my_config
#model.luke.config = my_config
model.eval()
# Load the tokenizer
#tokenizer = LukeTokenizer.from_pretrained("/mnt/shared/home/jose.luis.malaquias.ext/new_convertLUKE/LongLukeOpenEntity/Vanilla_long_3Agosto")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

logging.info("Data Memory after Loading Models")
#print_gpu_utilization()

logging.info("CHOOSE GPU")
########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0  # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
model = model.to(device)

batch_size = 128

num_predicted = 0
num_gold = 0
num_correct = 0

all_predictions = []
all_labels = []

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    num_gold += sum(len(l) for l in gold_labels)
    for logits, labels in zip(outputs.logits, gold_labels):
        for index, logit in enumerate(logits):
            if logit > 0:
                num_predicted += 1
                predicted_label = model.config.id2label[index]
                if predicted_label in labels:
                    num_correct += 1

precision = num_correct / num_predicted
recall = num_correct / num_gold
f1 = 2 * precision * recall / (precision + recall)

print(f"\n\nprecision: {precision} recall: {recall} f1: {f1}")

with open("results_OpenEntity_eval_LongLuke_5Aug_simple.txt", "w") as text_file:
    text_file.write(f"RESULTS \n precision: {precision} \n Recall: {recall} \n F1: {f1}")
