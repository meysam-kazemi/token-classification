# Named Entity Recognition (NER) Project

This project implements a Named Entity Recognition (NER) system using pre-trained models from the Hugging Face `transformers` library. The goal is to facilitate the identification and classification of named entities in text, such as people, organizations, locations, and other specific terms.


## Installation

To get started, ensure you have Python installed on your machine. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Train
### Download and Save the Dataset 

Run the script to create a local copy of the dataset.
```bash
python data/download_data.py
```

### Run the train.py
```bash
python src/train.py
```


## Inference

You can use this command:
```bash
python app.py
```

After executing the command above, you will be able to perform inference directly through a web interface.


