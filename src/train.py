from torch.utils.data import DataLoader
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

