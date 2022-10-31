from pynvml import *
import logging

from datasets import load_dataset
from datasets import ClassLabel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, TrainingArguments, Trainer
import torch
from tqdm import trange
# construir função que converta spans de relativos a frase para globais
import random
import os
import json

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


def load_examples_competition(dataset):
    examples = []


    for i, item in enumerate(dataset["test"]):
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
        aux_head = 0
        aux_tail = 0

        labels_pairs = []
        # get true labels
        for head_id in range(len(item["vertexSet"])):
            for tail_id in range(len(item["vertexSet"])):
                if (head_id!=tail_id):
                    labels_pair = tuple([head_id, tail_id , "Na"])
                    labels_pairs.append(labels_pair)

        entity_spans = [tuple(l) for l in entity_spans]
        oldToNewPos =  dict(zip(entity_spans, new_char_spans))
        entities = item["vertexSet"]
        correlations = []
        for pair in labels_pairs:
            head = random.choice(entities[pair[0]])
            tail = random.choice(entities[pair[1]])
            entity_head_id = pair[0]
            entity_tail_id = pair[1]
            rel = pair[2]

            if tuple(head["pos"]) in oldToNewPos:
                head["pos"]=oldToNewPos[tuple(head["pos"])]
            if tuple(tail["pos"]) in oldToNewPos:
                tail["pos"] = oldToNewPos[tuple(tail["pos"])]
            pack = tuple((head["pos"], tail["pos"], pair[2], tuple([entity_head_id, entity_tail_id]), item["title"]))

            item["vertexSet"] = entities
            examples.append(dict(
                text=text,
                entity_spans= pack[:2],
                labels = pack[2],
                idxs_entity_pair = pack[3],
                title = pack[4]
            ))
    return examples


torch.cuda.empty_cache()

dataset = load_dataset("docred")
max_value = 0
#for i, item in enumerate(dataset["train_annotated"]):
#    total_text_len = 0
#    tokens = item["sents"]
#    num_relations = len(item["labels"]["head"])

class ModifiedClassicLuke(LukeForEntityPairClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(in_features = 2048, out_features = 97, bias = True)

logging.info("Loading data and finetuned dataset for CLASSIC LUKE")
# FAZER LOAD DO MODEL FINETUNED DE 3 EPOCHS
model = ModifiedClassicLuke.from_pretrained("model_finetuned_classic")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

test_examples = load_examples_competition(dataset)
maximum = 0
max_seq = 0



logging.info("Memory before choosing GPU")
#torch.cuda.empty_cache()

########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0 # mudar para 0 para dar o cuda
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
    #gold_labels = [example["labels"] for example in batch_examples]
    idxs_entity_pair = [example["idxs_entity_pair"] for example in batch_examples]
    titles = [example["title"] for example in batch_examples]


for i in range(len(entity_spans)):
    entity_spans[i] = list(entity_spans[i])

del batch_examples

logging.info("Removing too big examples!!")
num_rejected = 0
clean_texts = []
clean_ents = []
clean_idxs_entity_pairs = []
clean_titles = []
tokenizer2 = LukeTokenizer.from_pretrained("studio-ousia/luke-large")
for ix in range(len(texts)):
    input = tokenizer2(texts[ix])
    if len(input.data["input_ids"]) > 500:
        num_rejected+=1
        continue
    clean_texts.append(texts[i])
    clean_ents.append(entity_spans[ix])
    clean_idxs_entity_pairs.append(idxs_entity_pair)
    clean_titles.append(titles)
texts = clean_texts
entity_spans = clean_ents
idxs_entity_pair = clean_idxs_entity_pairs
titles = clean_titles

torch.cuda.empty_cache()

relations_code_list = ["P1376",
                "P607",
                "P136",
                "P137",
                "P131",
                "P527",
                "P1412",
                "P206",
                "P205",
                "P449",
                "P127",
                "P123",
                "P86",
                "P840",
                "P355",
                "P737",
                "P740",
                "P190",
                "P576",
                "P749",
                "P112",
                "P118",
                "P17",
                "P19",
                "P3373",
                "P6",
                "P276",
                "P1001",
                "P580",
                "P582",
                "P585",
                "P463",
                "P676",
                "P674",
                "P264",
                "P108",
                "P102",
                "P25",
                "P27",
                "P26",
                "P20",
                "P22",
                "Na",
                "P807",
                "P800",
                "P279",
                "P1336",
                "P577",
                "P570",
                "P571",
                "P178",
                "P179",
                "P272",
                "P170",
                "P171",
                "P172",
                "P175",
                "P176",
                "P39",
                "P30",
                "P31",
                "P36",
                "P37",
                "P35",
                "P400",
                "P403",
                "P361",
                "P364",
                "P569",
                "P710",
                "P1344",
                "P488",
                "P241",
                "P162",
                "P161",
                "P166",
                "P40",
                "P1441",
                "P156",
                "P155",
                "P150",
                "P551",
                "P706",
                "P159",
                "P495",
                "P58",
                "P194",
                "P54",
                "P57",
                "P50",
                "P1366",
            "P1365",
            "P937",
            "P140",
            "P69",
            "P1198",
            "P1056"]

c2l = ClassLabel(num_classes = 97, names = relations_code_list)
label_list_ids = [c2l.str2int(label) for label in relations_code_list]
#gold_labels_ids = [c2l.str2int(label) for label in gold_labels]
#aa = [c2l.int2str(label) for label in gold_labels_ids] # convert ints to CODE of label!! USE IN EVAL


#inputs = tokenizer(text=texts[0], entity_spans = entity_spans[0], padding = "max_length", max_length = 1024, task = "entity_pair_classification", return_tensors = "pt")
#torch.save(inputs, 'inputs_eval.pt')
#test_dataset = MyDataset(inputs, gold_labels_ids)

logging.info("Beginning of evaluation batching")
output_dir = "evalClassic_17Out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, 'results.json')
output_file = open(output_filename, 'w')


batch_size = 10
rel2word = {
            "Na": "Na",
            "P6": "head of government",
            "P17": "country",
            "P19": "place of birth",
            "P20": "place of death",
            "P22": "father",
            "P25": "mother",
            "P26": "spouse",
            "P27": "country of citizenship",
            "P30": "continent",
            "P31": "instance of",
            "P35": "head of state",
            "P36": "capital",
            "P37": "official language",
            "P39": "position held",
            "P40": "child",
            "P50": "author",
            "P54": "member of sports team",
            "P57": "director",
            "P58": "screenwriter",
            "P69": "educated at",
            "P86": "composer",
            "P102": "member of political party",
            "P108": "employer",
            "P112": "founded by",
            "P118": "league",
            "P123": "publisher",
            "P127": "owned by",
            "P131": "located in the administrative territorial entity",
            "P136": "genre",
            "P137": "operator",
            "P140": "religion",
            "P150": "contains administrative territorial entity",
            "P155": "follows",
            "P156": "followed by",
            "P159": "headquarters location",
            "P161": "cast member",
            "P162": "producer",
            "P166": "award received",
            "P170": "creator",
            "P171": "parent taxon",
            "P172": "ethnic group",
            "P175": "performer",
            "P176": "manufacturer",
            "P178": "developer",
            "P179": "series",
            "P190": "sister city",
            "P194": "legislative body",
            "P205": "basin country",
            "P206": "located in or next to body of water",
            "P241": "military branch",
            "P264": "record label",
            "P272": "production company",
            "P276": "location",
            "P279": "subclass of",
            "P355": "subsidiary",
            "P361": "part of",
            "P364": "original language of work",
            "P400": "platform",
            "P403": "mouth of the watercourse",
            "P449": "original network",
            "P463": "member of",
            "P488": "chairperson",
            "P495": "country of origin",
            "P527": "has part",
            "P551": "residence",
            "P569": "date of birth",
            "P570": "date of death",
            "P571": "inception",
            "P576": "dissolved, abolished or demolished",
            "P577": "publication date",
            "P580": "start time",
            "P582": "end time",
            "P585": "point in time",
            "P607": "conflict",
            "P674": "characters",
            "P676": "lyrics by",
            "P706": "located on terrain feature",
            "P710": "participant",
            "P737": "influenced by",
            "P740": "location of formation",
            "P749": "parent organization",
            "P800": "notable work",
            "P807": "separated from",
            "P840": "narrative location",
            "P937": "work location",
            "P1001": "applies to jurisdiction",
            "P1056": "product or material produced",
            "P1198": "unemployment rate",
            "P1336": "territory claimed by",
            "P1344": "participant of",
            "P1365": "replaces",
            "P1366": "replaced by",
            "P1376": "capital of",
            "P1412": "languages spoken, written or signed",
            "P1441": "present in work",
            "P3373": "sibling"}


num_predicted = 0
num_gold = 0
num_correct = 0
this_pair = []
all_pairs = []
list_of_dicts = []
torch.cuda.empty_cache()

logging.info("Evaluation will start now!:")
model.eval()
model.to(device)
for batch_start_idx in trange(0, len(test_examples), batch_size):# len(test_examples) 100
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    idxs_entity_pair = [example["idxs_entity_pair"] for example in batch_examples]
    titles = [example["title"] for example in batch_examples]
    #gold_labels = [example["labels"] for example in batch_examples]

    #gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

    for i in range(len(entity_spans)):
        entity_spans[i] = list(entity_spans[i])


    inputs = tokenizer(text=texts, entity_spans=entity_spans, truncation=True, padding = "max_length", max_length = 512, task = "entity_pair_classification", return_tensors = "pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_indices = outputs.logits.argmax(-1)
    predicted_labels = [c2l.int2str(pred) for pred in predicted_indices.tolist()]
    predicted_relation = [rel2word.get(rel) for rel in predicted_labels]


    for i in range(len(predicted_relation)):
        list_of_dicts.append(dict(
            title=titles[i],
            h_idx=idxs_entity_pair[i][0],
            t_idx = idxs_entity_pair[i][1],
            r = predicted_relation[i]
        ))
    torch.cuda.empty_cache()

json_object = json.dumps(list_of_dicts, indent = 4)
with open("results_classic.json", "w") as outfile:
    outfile.write(json_object)
