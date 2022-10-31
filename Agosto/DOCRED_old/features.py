from datasets import load_dataset
from transformers import LukeTokenizer
import torch
from tqdm import trange

torch.cuda.empty_cache()

tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

max_pos = 1024
tokenizer.model_max_length = max_pos
tokenizer.init_kwargs['model_max_length'] = max_pos
model_path = "features_path"

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


def load_examples_train(dataset):
    examples = []


    for i, item in enumerate(dataset["train_annotated"]):
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
train_examples = load_examples_train(dataset)
new_list = []
for example in train_examples:
    for ix_rel in range(len(example["labels"])):
        new_entry = example.copy()
        new_entry["labels"] = new_entry["labels"][ix_rel]
        new_entry["entity_spans"] = new_entry["entity_spans"][ix_rel]
        new_list.append(new_entry)
        del new_entry
del train_examples
train_examples = []
train_examples = new_list
del new_list

torch.cuda.empty_cache()

########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0 # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")



# Convert to inputs
for batch_start_idx in trange(0, len(train_examples), len(train_examples)):
    batch_examples = train_examples[batch_start_idx:batch_start_idx+len(train_examples)]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["labels"] for example in batch_examples]

del train_examples

for i in range(len(entity_spans)):
    entity_spans[i] = list(entity_spans[i])

inputs = tokenizer(text=texts, entity_spans = entity_spans, padding = "max_length", max_length = 1024, task = "entity_pair_classification", return_tensors = "pt").to(device)
torch.save(inputs, 'inputs.pt')
