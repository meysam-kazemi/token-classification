# src/model.py

import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.utils import read_config


class modelTokenizer:
    def __init__(self):
        config = read_config()
        self.save_dir = config['model']['save_dir']
        self.model_name = config['model']['name']
        try:
            self._load_model_and_tokenizer_locally()
            print("Locally")
        except:
            self._load_and_save_model()
    def _load_and_save_model(self):
        """
        Loads a pre-trained model and tokenizer, and saves them to the specified directory.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Load the model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Save the model and tokenizer
        model.save_pretrained(self.save_dir)
        tokenizer.save_pretrained(self.save_dir)

        print(f"Model and tokenizer for '{self.model_name}' saved in '{self.save_dir}'.")

    def _load_model_and_tokenizer_locally(self):
        """
        Loads the model and tokenizer from the specified directory.
        """

        # Load the model and tokenizer from the specified directory
        self.model = AutoModelForTokenClassification.from_pretrained(self.save_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
