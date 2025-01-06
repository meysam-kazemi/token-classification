# src/utils.py
import os
import numpy as np
from datasets import load_dataset
import configparser
from transformers import DataCollatorForTokenClassification

def load_saved_dataset(dataset_name):
    """
    Loads the saved dataset from the specified directory.

    Args:
        dataset_name (str): The name of the dataset.
        data_dir (str): The directory where the dataset is saved.

    Returns:
        datasets.DatasetDict: Loaded dataset.
    """
    data_dir = 'data'
    # Load the dataset from the specified directory
    dataset = load_dataset('json', data_files={
        'train': os.path.join(data_dir, f"{dataset_name}_train.json"),
        'validation': os.path.join(data_dir, f"{dataset_name}_validation.json"),
        'test': os.path.join(data_dir, f"{dataset_name}_test.json")
    })
    
    return dataset


def read_config(config_file='config.ini'):
    """
    Reads the configuration file and returns the settings as a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    config = configparser.ConfigParser()
    
    # Check if the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file '{config_file}' does not exist.")
    
    config.read(config_file)
    
    # Convert config sections to a dictionary
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    
    return config_dict

class preProcessingTokens:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer      
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer
        )

    def _align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels 

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def tokenize_datasets(self, data):
        tokenized_datasets = data.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=data["train"].column_names,
        )
        print("Tokenized and labels aligned!")
        return tokenized_datasets
    
