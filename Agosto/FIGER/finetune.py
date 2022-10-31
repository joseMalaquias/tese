import load_model
import numpy as np
import torch.nn
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from tqdm import trange
from datasets import ClassLabel, load_dataset
import json
from datasets import Dataset, load_dataset
import logging
from transformers import BatchEncoding
from pynvml import *



def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


#def print_summary(result):
#    print(f"Time: {result.metrics['train_runtime']:.2f}")
#    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#    print_gpu_utilization()

logging.basicConfig(level=logging.INFO)
torch.cuda.empty_cache()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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

def preprocess_data(examples, num_examples):
    labels_matrix = np.zeros((num_examples, 112))
    #fill numpy_array
    for idx, label in enumerate(examples.labels):
        for item in range(len(label)):
            labels_matrix[idx,label[item]] +=1 #posicao a adicionar é label(item)
    examples.labels = labels_matrix.tolist()
    return examples

def tokenize_function(examples):
    return tokenizer(examples["texts"], entity_spans=examples["entity_spans"], padding="max_length")
train_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/Agosto/FIGER/data/train.json")

print("\n\n############ Prepare finetune parameters to feed model ##################")
label_list = ['/art', '/art/film', '/astral_body', '/award', '/biology', '/body_part', '/broadcast', '/broadcast/tv_channel', '/broadcast_network', '/broadcast_program', '/building', '/building/airport', '/building/hospital', '/building/library', '/building/restaurant', '/building/sports_facility', '/building/theater', '/chemistry', '/computer', '/computer/programming_language', '/disease', '/education', '/education/department', '/education/educational_degree', '/event', '/event/attack', '/event/election', '/event/military_conflict', '/event/natural_disaster', '/event/protest', '/event/sports_event', '/finance', '/finance/currency', '/food', '/game', '/geography', '/geography/island', '/geography/mountain', '/god', '/government', '/government/government', '/government/political_party', '/government_agency', '/internet', '/internet/website', '/language', '/law', '/livingthing', '/livingthing/animal', '/location', '/location/body_of_water', '/location/bridge', '/location/cemetery', '/location/city', '/location/country', '/location/county', '/location/province', '/medicine', '/medicine/drug', '/medicine/medical_treatment', '/medicine/symptom', '/metropolitan_transit', '/metropolitan_transit/transit_line', '/military', '/music', '/news_agency', '/organization', '/organization/airline', '/organization/company', '/organization/educational_institution', '/organization/fraternity_sorority', '/organization/sports_league', '/organization/sports_team', '/park', '/people', '/people/ethnicity', '/person', '/person/actor', '/person/architect', '/person/artist', '/person/athlete', '/person/author', '/person/coach', '/person/director', '/person/doctor', '/person/engineer', '/person/monarch', '/person/musician', '/person/politician', '/person/religious_leader', '/person/soldier', '/play', '/product', '/product/airplane', '/product/car', '/product/computer', '/product/instrument', '/product/ship', '/product/spacecraft', '/product/weapon', '/rail', '/rail/railway', '/religion', '/religion/religion', '/software', '/time',  '/title', '/train', '/transit', '/transportation', '/transportation/road', '/written_work']
c2l = ClassLabel(num_classes = 112, names = label_list)
label_list_ids = [c2l.str2int(label) for label in label_list]


logging.info("Data Memory before Loading Models")
#print_gpu_utilization()


logging.info("LOADING MODEL ..........")
#model = load_model.model
tokenizer = load_model.tokenizer
model_path = load_model.model_path


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

logging.info(f"Which device is used?: {device}")

logging.info("Extract from raw text to each field")
batch_size = 1 #4
first = 1
all_gold_labels_ids = []
all_inputs = BatchEncoding()
max_pos = load_model.max_pos
torch.cuda.empty_cache()
# extract content of file to each one of the fields
for batch_start_idx in trange(0, len(train_examples), batch_size):
    batch_examples = train_examples[batch_start_idx:batch_start_idx+ batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]
    inputs = tokenizer(texts, entity_spans = entity_spans, padding = "max_length", max_length = max_pos, return_tensors = "pt").to(device)
    gold_labels_ids = [c2l.str2int(label) for label in gold_labels]
    torch.cuda.empty_cache()
    if first == 1:
        all_inputs = inputs
        all_gold_labels_ids+=gold_labels_ids
        first = 0
    else:
        all_inputs.data['input_ids'] = torch.cat([all_inputs.data['input_ids'], inputs.data['input_ids']], dim = 0)
        all_inputs.data['entity_ids'] = torch.cat([all_inputs.data['entity_ids'], inputs.data['entity_ids']], dim = 0)
        all_inputs.data['entity_position_ids'] = torch.cat([all_inputs.data['entity_position_ids'], inputs.data['entity_position_ids']], dim = 0)
        all_inputs.data['attention_mask'] = torch.cat([all_inputs.data['attention_mask'], inputs.data['attention_mask']], dim = 0)
        all_inputs.data['entity_attention_mask'] = torch.cat([all_inputs.data['entity_attention_mask'], inputs.data['entity_attention_mask']], dim = 0)
        del inputs
        all_gold_labels_ids+=gold_labels_ids
        del gold_labels_ids
    torch.cuda.empty_cache()

logging.info("\n SAVING inputs to tensor..........")
torch.save(all_inputs, 'inputs.pt')
del all_inputs
logging.info("main inputs are saved")
torch.save(all_gold_labels_ids, 'all_gold_labels_ids.pt')
torch.cuda.empty_cache()
logging.info("final memory")
print_gpu_utilization()


"""
inputs.to(device)
inputs.cuda()
#inputs = torch.load("all_inputs.pt")
#gold_labels = torch.load("all_gold_labels_ids.pt")
train_dataset = MyDataset(inputs, gold_labels_ids)
train_dataset_ready = preprocess_data(train_dataset, len(texts))

logging.info("Data Memory before training")
#print_gpu_utilization()
model = model.to(device)

training_args = TrainingArguments(
    do_train=True,
    per_device_train_batch_size=batch_size,
    output_dir="./LongFIGER_finetuned",
    metric_for_best_model="f1",
    gradient_accumulation_steps=4,
    gradient_checkpointing = True,
    fp16 = True,
    optim = "adafactor",
    learning_rate = 0.00003
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_ready,
)

torch.cuda.memory_summary(device=None, abbreviated=False)

logging.info("TRAIN WILL BEGIN!!......")
result = trainer.train()
print_summary(result)
logging.info("SAVING MODEL")
model.save_pretrained(save_directory="./LongFIGER_finetuned")
tokenizer.save_pretrained(save_directory="./LongFIGER_finetuned")
# to load this saved model we use:
# model = .from_pretrained("path/to/awesome-name-you-picked").
print("END of TRAINING ! !")

#outputs = model(**inputs)
#logits = outputs.logits
#logits = logits.detach().numpy()
#predicted_class_id=logits[0].argmax(-1).item()
#print("Predicted class:", model.config.id2label[predicted_class_id])

"""





