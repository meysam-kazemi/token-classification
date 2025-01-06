# src/classifier.py

from transformers import pipeline
from utils import read_config

config = read_config()
model_checkpoint = config['model']['save_dir']
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
def tabluar_result(res):
    output_text = ""
    words = [ent['word'].center(10) for ent in res]
    groups = [ent['entity_group'].center(10) for ent in res]
    scores = [(str(round(ent['score']*100,1))+"%").center(10) for ent in res]
    output_text += "words   :  " + "|".join(words) + "\n"
    output_text += "groups  :  " + "|".join(groups) + "\n"
    output_text += "scores  :  " + "|".join(scores) + "\n"
    return output_text


if __name__=="__main__":
    input_text = ""
    while 1:
        input_text = input("please write a text:\n\t")
        if input_text=="q":
            break
        res = token_classifier(input_text)

        print(f"the result is :")
        print(tabluar_result(res))
