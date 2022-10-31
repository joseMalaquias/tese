# Document that evaluates DOCRED with LongLUKE. Loads model from model_finetuned_3epocs
import logging

from datasets import load_dataset
from datasets import ClassLabel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, TrainingArguments, Trainer
import torch
from tqdm import trange
# construir função que converta spans de relativos a frase para globais
import load_model
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def convert_spans(item):
    sents = []
    sent_map = []

    entities = item["vertexSet"]
    entity_start, entity_end = [], []
    mention_types = []
    entity_spans = []
    for entity in entities:
        for mention in entity:
            if mention["sent_id"] != 0:
                current_id = mention["sent_id"]
                mention["pos"] = [sum(len(s) for s in item["sents"][:current_id])+mention["pos"][0],
                    sum(len(s) for s in item["sents"][:current_id])+mention["pos"][1]]
                mention["sent_id"] = 0

            pos = mention["pos"]
            mention_types.append(mention['type'])
            entity_spans.append(pos)
    item["vertexSet"] = entities
    return item, entity_spans





def load_examples_test(dataset):
    examples = []


    for i, item in enumerate(dataset["validation"]):
        concat_tokens = []
        counter = 0
        converted_item, entity_spans = convert_spans(item)
        tokens = item["sents"]
        for j in range(len(tokens)):
            concat_tokens += tokens[j]
        del j
        tokens = concat_tokens
        del concat_tokens

        # new
        text = ""
        cur = 0
        new_char_spans = [0]*len(entity_spans)
        entity_spans.sort(key=lambda y:y[0])
        for target_entity in entity_spans:
            tamanho_texto = len(text)
            text += " ".join(tokens[cur: target_entity[0]])
            if text:
                text += " "
            char_start = len(text)
            text += " ".join(tokens[target_entity[0]: target_entity[1]])
            char_end = len(text)
            new_char_spans[counter] = (char_start, char_end)
            text += " "
            cur = target_entity[1]
            counter+=1
        text += " ".join(tokens[cur:])
        text = text.rstrip()
        # get true labels
        labels_pairs = tuple(zip(item["labels"]["head"], item["labels"]["tail"], item["labels"]["relation_id"]))
        entity_spans = [tuple(l) for l in entity_spans]
        oldToNewPos =  dict(zip(entity_spans, new_char_spans))
        entities = item["vertexSet"]
        correlations = []
        for pair in labels_pairs:
            for head in entities[pair[0]]:
                if tuple(head["pos"]) in oldToNewPos:
                    head["pos"]=oldToNewPos[tuple(head["pos"])]
                for tail in entities[pair[1]]:
                    if tuple(tail["pos"]) in oldToNewPos:
                        tail["pos"] = oldToNewPos[tuple(tail["pos"])]
                    pack = tuple((head["pos"], tail["pos"], pair[2]))
                    correlations += (pack),
        item["vertexSet"] = entities
        examples.append(dict(
            text=text,
            entity_spans= [d[:][:-1] for d in correlations],
            labels = [d[:][-1] for d in correlations]
        ))
    return examples

dataset = load_dataset("docred")
max_value = 0
#for i, item in enumerate(dataset["train_annotated"]):
#    total_text_len = 0
#    tokens = item["sents"]
#    num_relations = len(item["labels"]["head"])


# FAZER LOAD DO MODEL FINETUNED DE 3 EPOCHS
model = load_model.model
tokenizer = load_model.tokenizer
test_examples = load_examples_test(dataset)



new_list = []
for example in test_examples:
    for ix_rel in range(len(example["labels"])):
        new_entry = example.copy()
        new_entry["labels"] = new_entry["labels"][ix_rel]
        new_entry["entity_spans"] = new_entry["entity_spans"][ix_rel]
        new_list.append(new_entry)
        del new_entry
del test_examples
test_examples = []
test_examples = new_list
del new_list

logging.info("Memory before choosing GPU")
torch.cuda.empty_cache()
print_gpu_utilization()

########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
#model = model.to(device)
#model.eval()


# Convert to inputs
for batch_start_idx in trange(0, len(test_examples), len(test_examples)):
    batch_examples = test_examples[batch_start_idx:batch_start_idx+len(test_examples)]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["labels"] for example in batch_examples]

for i in range(len(entity_spans)):
    entity_spans[i] = list(entity_spans[i])

del batch_examples


c2l = ClassLabel(num_classes = 96, names = model.config.relations_code_list)
label_list_ids = [c2l.str2int(label) for label in model.config.relations_code_list]
gold_labels_ids = [c2l.str2int(label) for label in gold_labels]


logging.info("Beginning of evaluation batching")
torch.cuda.empty_cache()
print_gpu_utilization()

batch_size = 8

num_predicted = 0
num_gold = 0
num_correct = 0

model.eval()
model.to(device)

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["labels"] for example in batch_examples]

    gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

    for i in range(len(entity_spans)):
        entity_spans[i] = list(entity_spans[i])

    inputs = tokenizer(text = texts, entity_spans=entity_spans, return_tensors="pt", padding="max_length", max_length = 1024, task = "entity_pair_classification").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_indices = outputs.logits.argmax(-1)
    predicted_labels = [c2l.int2str(pred) for pred in predicted_indices.tolist()]
    for predicted_label, gold_label in zip(predicted_labels, gold_labels):
        if predicted_label != "no_relation":
            num_predicted += 1
        if gold_label != "no_relation":
            num_gold += 1
            if predicted_label == gold_label:
                num_correct += 1

    torch.cuda.empty_cache()

precision = num_correct / num_predicted
recall = num_correct / num_gold
f1 = 2 * precision * recall / (precision + recall)
del test_examples

logging.info("END OF EVALUATION. RESULTS ARE NOW BEING SAVED...!\n")
with open("results_long_DOCRED_finetuned.txt", "w") as text_file:
    text_file.write(f"RESULTS \n num_correct = {num_correct} \n num_predicted = {num_predicted} \n num_gold = {num_gold} \n precision: {precision} \n Recall: {recall} \n f1: {f1}")

logging.info("END OF PROGRAM ! ! !")


