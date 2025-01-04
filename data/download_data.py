# src/download_data

import os
from datasets import load_dataset

def download_dataset(dataset_name, save_dir):
    """
    Downloads the specified dataset and saves it to the given directory.

    Args:
        dataset_name (str): The name of the dataset to download.
        save_dir (str): The directory where the dataset will be saved.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Save the dataset to the specified directory
    for split in dataset.keys():
        dataset[split].to_json(os.path.join(save_dir, f"{dataset_name}_{split}.json"))

    print(f"Dataset '{dataset_name}' downloaded and saved in '{save_dir}'.")

if __name__ == "__main__":
    # Example usage
    dataset_name = "conll2003"  # Change this to your desired dataset
    download_dataset(dataset_name, 'data')