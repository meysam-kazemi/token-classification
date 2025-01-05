from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from model import modelTokenizer
from utils import (
    read_config,
    load_saved_dataset,
    preProcessingTokens,
    postprocess,
)


config = read_config()
data = load_saved_dataset(config['dataset']['name'])
mt = modelTokenizer(config)
preprocess = preProcessingTokens(tokenizer=mt.tokenizer)
tokenized_datasets = preprocess.tokenize_datasets(data)



train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=preprocess.data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=preprocess.data_collator, batch_size=8
)

optimizer = AdamW(mt.model.parameters(), lr=2e-5)


accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    mt.model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
