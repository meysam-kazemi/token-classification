# src/model.py

import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import read_config

def load_and_save_model(model_name, save_dir):
    """
    Loads a pre-trained model and tokenizer, and saves them to the specified directory.

    Args:
        model_name (str): The name of the pre-trained model to load (e.g., 'bert-base-uncased').
        save_dir (str): The directory where the model and tokenizer will be saved.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Model and tokenizer for '{model_name}' saved in '{save_dir}'.")


def load_model_and_tokenizer_locally(model_dir):
    """
    Loads the model and tokenizer from the specified directory.

    Args:
        model_dir (str): The directory where the model and tokenizer are saved.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    # Load the model and tokenizer from the specified directory
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer

if __name__=="__main__":
    config = read_config()
    # load_and_save_model(
    #     model_name=config['model']['name'],
    #     save_dir=config['model']['save_dir']
    #     )
    model, tokenizer = load_model_and_tokenizer_locally(
        config['model']['save_dir']
    )
    print(tokenizer.is_fast)