# src/model.py

import os
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import evaluate
from src.utils import read_config


class modelTokenizer:
    def __init__(self, label_names):
        config = read_config()
        self.save_dir = config['model']['save_dir']
        self.model_name = config['model']['name']
        try:
            self._load_model_and_tokenizer_locally()
            print("Locally")
        except:
            self._load_and_save_model()
        
        self.label_names = label_names
        self.eval = evaluate.load("seqeval")
        self.id2label = {i: label for i, label in enumerate(label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}


    def _load_and_save_model(self):
        """
        Loads a pre-trained model and tokenizer, and saves them to the specified directory.
        """
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Load the model and tokenizer
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Save the model and tokenizer
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)

        print(f"Model and tokenizer for '{self.model_name}' saved in '{self.save_dir}'.")

    def _load_model_and_tokenizer_locally(self):
        """
        Loads the model and tokenizer from the specified directory.
        """

        # Load the model and tokenizer from the specified directory
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.save_dir,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.eval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }