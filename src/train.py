from transformers import TrainingArguments, Trainer
from model import modelTokenizer
from utils import (
    read_config,
    load_saved_dataset,
    preProcessingTokens,
)

config = read_config()
data = load_saved_dataset(config['dataset']['name'])
mt = modelTokenizer(config)
preprocess = preProcessingTokens(tokenizer=mt.tokenizer)
tokenized_datasets = preprocess.tokenize_datasets(data)

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)




trainer = Trainer(
    model=mt.model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=preprocess.data_collator,
    compute_metrics=mt.compute_metrics,
    tokenizer=mt.tokenizer,
)
trainer.train()
