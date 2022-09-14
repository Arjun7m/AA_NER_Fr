'''
Configuration:
# - Model: wiki-4gb-wiki
# - Source data: europenan train , dev, test
# - Target data: mixed_fr_train with all tags as 'O'

# Note: Make sure your version of Transformers is at least 4.11.0 since the functionality was introduced in that version:
'''
# Dependencies:
from pynvml import *
from tqdm.auto import tqdm
from rich.progress import track
import random
from IPython.display import display, HTML
import pandas as pd
import numpy as np

from torchinfo import summary
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.autograd import Function

from datasets import ClassLabel, Sequence, load_metric, load_dataset
import datasets
import transformers
from transformers import get_scheduler
from transformers import DataCollatorForTokenClassification
from transformers import AutoConfig, AutoTokenizer ,PreTrainedModel, CamembertModel, CamembertConfig
from transformers.modeling_outputs import TokenClassifierOutput


source_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "ner_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=['I-PER', 'O', 'I-LOC', 'I-ORG']
            )
        ),
    }
)

source_datasets = load_dataset('json', data_files={'train': 'datasets/euro_train.json',
                                                'validation': 'datasets/euro_dev.json',
                                                'test': 'datasets/euro_test.json'},
                            features=source_features, field='data')


target_features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "tokens": datasets.Sequence(datasets.Value("string")),
        "ner_tags": datasets.Sequence(
            datasets.features.ClassLabel(
                names=['O']
            )
        ),
    }
)
target_datasets = load_dataset('json', data_files={'train': '/content/gdrive/MyDrive/version 5/French Json data/mixed_fr_train.json'},
                        features=target_features, field='data')

'''
Note: The labels are  encoded as integer ids to be easily usable by our model, but the correspondence with the actual categories is stored in the `features` of the dataset:

So for the NER tags, 0 corresponds to 'O', 1 to 'B-PER' etc... On top of the 'O' (which means no special entity), there are four labels for NER here, each prefixed with 'B-' (for beginning) or 'I-' (for intermediate), that indicate if the token is the first one for the current group with the label or not:
# - 'PER' for person
# - 'ORG' for organization
# - 'LOC' for location
# - 'MISC' for miscellaneous
'''

task = "ner"
label_list = source_datasets["train"].features[f"{task}_tags"].feature.names
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

# For Source Domain Training Set
show_random_elements(source_datasets["train"])

# For Target Domain Training Set
show_random_elements(target_datasets["train"])


# Preprocessing the data and loading the model
tokenizer_checkpoint = "camembert-base"
model_checkpoint = "camembert/camembert-base-wikipedia-4gb"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# # Note : transformers are often pretrained with subword tokenizers, meaning that even if your inputs have been split into words already, each of those words could be split again by the tokenizer. Let's look at an example of that:
# # Code to check difference in length of tokenized inputs per sentence and their respective labels:

# # For Source Dataset
# faulty_train = []
# for i in range(len(source_datasets['train'])):
#     example = source_datasets['train'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_train.append(i)

# faulty_dev = []
# for i in range(len(source_datasets['validation'])):
#     example = source_datasets['validation'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_dev.append(i)

# faulty_test = []
# for i in range(len(source_datasets['test'])):
#     example = source_datasets['test'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_test.append(i)

# print(
#     f'Faulty Sentences in Source training set : {faulty_train} \n Faulty Sentences in Source validation set : {faulty_dev} \n Faulty Sentences in Source  testing set : {faulty_test} \n ')

# # For Target Dataset
# faulty_train = []
# for i in range(len(target_datasets['train'])):
#     example = target_datasets['train'][i]
#     if len(example['tokens']) != len(example['ner_tags']):
#         faulty_train.append(i)

# print(f'Faulty Sentences in training set : {faulty_train}')


# This function returns the final encoded labels for the training set with length accounted for before and after tokenization changes to individual tokens
label_all_tokens = True
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)

    labels = []
    i = 0
    for label in examples[f"{task}_tags"]:
        # print(f'sentence no {i} is good')
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                # print(f' {word_idx} index went well')
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
        i += 1
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Model Architecture
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class CamembertForTokenClassification(PreTrainedModel):
    config_class = CamembertConfig

    def __init__(self, bert, config):
        super(CamembertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        # Load model body
        self.bert = bert
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Domain Classifier:
        self.domain_classifier = nn.Sequential(
            nn.Linear(2*config.hidden_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, grl_lambda=1.0, **kwargs):
        # Use model body to get encoder representations
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, output_hidden_states=True, **kwargs)

        # Apply classifier to encoder representation
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)

        # domain Classifier:
        hidden_states = outputs[2]
        pooled_output = torch.cat(
            tuple([hidden_states[i] for i in [-2, -1]]), dim=-1)
        pooled_output = pooled_output[:, 0, :]
        cls_output = self.dropout(pooled_output)
        reversed_cls_output = GradientReversalFn.apply(cls_output, grl_lambda)
        domain_pred = self.domain_classifier(reversed_cls_output)

        loss_ner = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_ner = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss_ner, logits=logits,
                                    hidden_states=outputs.hidden_states,
                                    attentions=outputs.attentions), domain_pred

# Model Declarations
camembert_config = AutoConfig.from_pretrained(model_checkpoint,
                                            num_labels=len(label_list), output_hidden_states=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = CamembertModel.from_pretrained(model_checkpoint, add_pooling_layer=True)
model = (CamembertForTokenClassification(bert, config=camembert_config)
        .to(device))
print(model)
summary(model)

# Data Loading using standard Pytorch
data_collator = DataCollatorForTokenClassification(tokenizer)

# To apply this function on all the sentences (or pairs of sentences) in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.
# Source dataset
s_tokenized_datasets = source_datasets.map(
    tokenize_and_align_labels, batched=True, batch_size=1000, remove_columns=['id', 'tokens', 'ner_tags'])
print('Source datasets are tokenized !!!\n')

# Target dataset
t_tokenized_datasets = target_datasets.map(
    tokenize_and_align_labels, batched=True, batch_size=1000, remove_columns=['id', 'tokens', 'ner_tags'])
print('\nTarget datasets are tokenized !!!\n')

#Source Dataloading:
batch_size = 16
s_train_dataloader = DataLoader(
    s_tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
s_eval_dataloader = DataLoader(
    s_tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
)

#Target Dataloading:
batch_size = 16
t_train_dataloader = DataLoader(
    t_tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# Hyper-Parameters
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 16
num_training_steps = num_epochs * len(s_train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
loss_fxn_domain = nn.NLLLoss()
metric = load_metric("seqeval")

# Model Checkpoint Logging
checkpoint_dir = './Camembert-4g-wiki_da/inference/'
model_dir = '/Camembert-4g-wiki_da/best_model/'
inference_dir = './Camembert-4g-wiki_da/inference/'
model_name = model_checkpoint.split('/')[0]

def save_ckp(state, checkpoint_dir, best_model_dir, is_best=False):
    # General Saving whole model for training with epoch as well
    f_path = checkpoint_dir + f'{model_name}-checkpoint.pth'
    torch.save(state, f_path)

    # Saving model for inference only
    inf_path = inference_dir + f'{model_name}-checkpoint-{state["epoch"]}.pth'
    torch.save(state['state_dict'], inf_path)

    if is_best:
        # Saving model for inference only
        inf_path = inference_dir + f'{model_name}-checkpoint.pth'
        torch.save(state['state_dict'], inf_path)

# Training Code
start_epoch = 1
results_recall = []
results_precision = []
results_f1_score = []
results_accuracy = []
epoch_list = []
max_batches = min(len(s_train_dataloader), len(t_train_dataloader))
num_labels = len(label_list)
max_f1_score = 0

for epoch in range(start_epoch, num_epochs+1):
    print(f'\nTraining Going on for epoch : {epoch}')
    source_iterator = iter(s_train_dataloader)
    target_iterator = iter(t_train_dataloader)

    for batch_idx in track(range(max_batches), description=f'Training for {epoch} :', total=max_batches):
        p = float(batch_idx + epoch * max_batches) / (num_epochs * max_batches)
        grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
        grl_lambda = torch.tensor(grl_lambda)

        model.train()
        batch_s = next(source_iterator)
        batch = {k: v.to(device) for k, v in batch_s.items()}
        batch['grl_lambda'] = grl_lambda
        outputs, domain_pred = model(**batch)

        labels = batch['labels']
        logits = outputs.logits
        loss_ner = outputs.loss

        # Domain Losses
        y_s_domain = torch.zeros(batch_size, dtype=torch.long).to(device)
        loss_s_domain = loss_fxn_domain(domain_pred, y_s_domain)

        batch_t = next(target_iterator)
        batch = {k: v.to(device) for k, v in batch_t.items()}
        batch['grl_lambda'] = grl_lambda
        _, domain_pred = model(**batch)

        # Domain Losses
        y_t_domain = torch.ones(batch_size, dtype=torch.long).to(device)
        loss_t_domain = loss_fxn_domain(domain_pred, y_t_domain)

        loss = loss_ner + 2*(loss_s_domain + loss_t_domain)
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
    for batch in track(s_eval_dataloader, total=len(s_eval_dataloader), description='Validating...'):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs, _ = model(**batch)
        logits = outputs.logits
        output = torch.argmax(logits, dim=-1)

        predictions, labels = output, batch['labels']
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)]
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute(
        predictions=true_predictions, references=true_labels)

    print(f'\nThe Validation result for epoch {epoch} is:\n')

    if (results["overall_f1"]*100) > max_f1_score:
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
s_test_dataloader = DataLoader(
    s_tokenized_datasets["test"], batch_size=16, collate_fn=data_collator)

# Specify a path
inference_dir = './Camembert-4g-wiki_da/inference/'
inf_path = inference_dir + f'{model_name}-checkpoint.pth'
# Loading model for eval
model.load_state_dict(torch.load(inf_path))

model.eval()
for batch in track(s_test_dataloader, description='Testing...', total=len(s_test_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs, _ = model(**batch)
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

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)


# *******************************END*********************************************


