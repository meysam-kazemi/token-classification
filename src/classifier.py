# src/classifier.py

from transformers import pipeline
from utils import read_config

config = read_config()
model_checkpoint = config['model']['save_dir']
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)

input_text = ""
while 1:
    input_text = input("please write a text:\n\t")
    if input_text=="q":
        break
    res = token_classifier(input_text)

    print(f"the result is :")
    words = [ent['word'].center(10) for ent in res]
    print("words   :  " + "|".join(words))
    groups = [ent['entity_group'].center(10) for ent in res]
    print("groups  :  " + "|".join(groups))
    scores = [(str(round(ent['score']*100,1))+"%").center(10) for ent in res]
    print("scores  :  " + "|".join(scores))
