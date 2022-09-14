'''
Configuration:
- Model: Flaubert
- tokenizer: Flaubert
- data: europena train, dev, test

# Note: Make sure your version of Transformers is at least 4.11.0 since the functionality was introduced in that version:
'''
# Dependencies:
from pynvml import *
from rich.progress import track
from tqdm.auto import tqdm
from IPython.display import display, HTML
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim import AdamW
from torch.utils.data import DataLoader

import datasets
from datasets import ClassLabel, Sequence, load_dataset, load_metric
import transformers
from transformers import get_scheduler
from transformers import AutoConfig, PreTrainedModel,FlaubertConfig, AutoConfig, AutoModel, FlaubertModel, AutoTokenizer
from torch.autograd import Function
from transformers.modeling_outputs import TokenClassifierOutput



# Loading our custom dataset:
features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "ner_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=['I-PER', 'O', 'I-LOC', 'I-ORG'],
            )
        ),
    }
)


datasets = load_dataset('json', data_files={'train': './euro_train.json',
                                            'validation': './euro_dev.json',
                                            'test': './euro_test.json'},
                        features=features, field='data')


'''
Note: The labels are  encoded as integer ids to be easily usable by our model, but the correspondence with the actual categories is stored in the `features` of the dataset:

So for the NER tags, 0 corresponds to 'O', 1 to 'B-PER' etc... On top of the 'O' (which means no special entity), there are four labels for NER here, each prefixed with 'B-' (for beginning) or 'I-' (for intermediate), that indicate if the token is the first one for the current group with the label or not:
# - 'PER' for person
# - 'ORG' for organization
# - 'LOC' for location
# - 'MISC' for miscellaneous
'''

# Since the labels are lists of `ClassLabel`, the actual names of the labels are nested in the `feature` attribute of the object above:
task = "ner"
label_list = datasets["train"].features[f"{task}_tags"].feature.names
print(label_list)


# To get a sense of what the data looks like, the following function will show some examples picked randomly in the dataset (automatically decoding the labels in passing).
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(
        dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(
                lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))
    
show_random_elements(datasets["train"])


# Preprocessing the data and loading the model
tokenizer_checkpoint = "flaubert/flaubert_base_uncased"
model_checkpoint = "flaubert/flaubert_base_uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

assert isinstance(
    tokenizer, transformers.PreTrainedTokenizerFast), 'Yes It is a Fast Tokenizer Instance'

# # Note : transformers are often pretrained with subword tokenizers, meaning that even if your inputs have been split into words already, each of those words could be split again by the tokenizer. Let's look at an example of that:
# # Code to check difference in length of tokenized inputs per sentence and their respective labels:
# faulty_train = []
# for i in range(len(datasets['train'])):
#     example = datasets['train'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_train.append(i)

# faulty_dev = []
# for i in range(len(datasets['validation'])):
#     example = datasets['validation'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_dev.append(i)

# faulty_test = []
# for i in range(len(datasets['test'])):
#     example = datasets['test'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_test.append(i)

# print(
#     f'Faulty Sentences in training set : {faulty_train} \n Faulty Sentences in validation set : {faulty_dev} \n Faulty Sentences in testing set : {faulty_test} \n ')


# This function returns the final encoded labels for the training set with length accounted for before and after tokenization changes to individual tokens
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)

    labels = []
    for i in range(len(tokenized_inputs['input_ids'])):
        # one sentence at a time from the dataset
        example = examples['tokens'][i]
        final_tokens = ['<s>'] + []

        tags = examples['ner_tags'][i] + [-100]
        final_tags = [-100]
        for j, word in enumerate(example):
            tokens = tokenizer(" " + word, truncation=True, max_length=512)
            # final_tokens.append(tokenizer.convert_ids_to_tokens(tokens['input_ids'])[1])
            temp = tokenizer.convert_ids_to_tokens(tokens['input_ids'])[1:-1]
            for inp in temp:
                final_tokens.append(inp)
                final_tags.append(tags[j])
        final_tags = final_tags + [-100]
        labels.append(final_tags)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Model Architecture
class FlaubertForTokenClassification(PreTrainedModel):
    config_class = FlaubertConfig
    def __init__(self, bert, config):
        super(FlaubertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        # Load model body
        self.bert = bert
        # Set up token classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.emb_dim, config.num_labels)
        # Load and initialize weights
        # self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                labels=None, **kwargs):
        # Use model body to get encoder representations
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, output_hidden_states = True ,**kwargs)

        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Return model output object
        return TokenClassifierOutput(loss=loss, logits=logits, 
                                    hidden_states=outputs.hidden_states, 
                                    attentions=outputs.attentions)


# Model Declaration
flaubert_config = AutoConfig.from_pretrained(model_checkpoint, 
                                        num_labels=len(label_list), output_hidden_states = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlaubertForTokenClassification(AutoModel.from_pretrained(model_checkpoint),config=flaubert_config).to(device)
print(model)
summary(model)

# Data Loading using standard Pytorch
data_collator = DataCollatorForTokenClassification(tokenizer)

# To apply this function on all the sentences (or pairs of sentences) in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True,
                                batch_size=1000, remove_columns=['id', 'tokens', 'ner_tags'])
tokenized_datasets['train'].column_names


batch_size = 16
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# Hyoer-parameters
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 8
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
metric = load_metric("seqeval")


# Model Checkpoints logging
checkpoint_dir = './flaubert_non-da/'
model_dir = './flaubert_non-da/best_model/'
inference_dir = './flaubert_non-da/inference/'
model_name = model_checkpoint.split('/')[0]


def save_ckp(state, checkpoint_dir, best_model_dir, is_best=False):
    # General Saving whole model for training with epoch as well
    f_path = checkpoint_dir + f'{model_name}-checkpoint-{state["epoch"]}.pth'
    torch.save(state, f_path)

    # Saving model for inference only
    inf_path = inference_dir + f'{model_name}-checkpoint-{state["epoch"]}.pth'
    torch.save(state['state_dict'], inf_path)

    if is_best:
        # Saving model for inference only
        inf_path = inference_dir + f'{model_name}-checkpoint.pth'
        torch.save(state['state_dict'], inf_path)


# Training Code:
start_epoch = 1
max_f1_score = 0
results_recall = []
results_precision = []
results_f1_score = []
results_accuracy = []
epoch_list = []

for epoch in range(start_epoch, num_epochs+1):
    print(f'\nTraining for epoch : {epoch}\n')
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}

    # Get GPU Stats while training
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'\nCurrent Status of GPU after Epoch : {epoch}\n')
    print(f'total    : {info.total/(1024*1024)} Mib')
    print(f'free     : {info.free/(1024*1024)} Mib')
    print(f'used     : {info.used/(1024*1024)} Mib')

    model.eval()
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        output = torch.argmax(logits, dim=-1)

        predictions, labels = output, batch['labels']

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute(
        predictions=true_predictions, references=true_labels)
    print(f'\nThe Validation result for epoch {epoch+1} is:\n')

    if results["overall_f1"]*100 > max_f1_score:
        max_f1_score = results["overall_f1"]*100

    print({
        "precision": results["overall_precision"]*100,
        "recall": results["overall_recall"]*100,
        "f1": results["overall_f1"]*100,
        "accuracy": results["overall_accuracy"]*100,
    }, '\n')

    results_precision.append(results["overall_precision"]*100)
    results_recall.append(results["overall_recall"]*100)
    results_f1_score.append(results["overall_f1"]*100)
    results_accuracy.append(results["overall_accuracy"]*100)
    epoch_list.append(epoch)

    if max_f1_score >= results["overall_f1"]*100:
        is_best = True

    save_ckp(checkpoint, checkpoint_dir, model_dir, is_best)


# Training Results in Tabular Format
recall, precision, f1_score, accuracy, epochs = results_recall, results_precision, results_f1_score, results_accuracy, epoch_list

df = pd.DataFrame(list(zip(epochs, f1_score, recall, precision, accuracy)),
                columns=['epoch', 'f1-score', 'recall', 'precision', 'accuracy'])

df.sort_values(['f1-score'], ascending=False)
print(df)


# Test Set Evaluation Code:
test_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=16, collate_fn=data_collator
)

# Specify a path
model_name = model_checkpoint.split('/')[0]
inference_dir = './flaubert_non-da/inference/'
inf_path = inference_dir + f'{model_name}-checkpoint.pth'

# Loading model for eval
model.load_state_dict(torch.load(inf_path))

model.eval()
for batch in tqdm(test_dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    output = torch.argmax(logits, dim=-1)

    predictions, labels = output, batch['labels']
    # predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    metric.add_batch(predictions=true_predictions, references=true_labels)

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)

# *******************************END*********************************************
