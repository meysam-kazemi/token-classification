# src/classifier.py
from transformers import pipeline
from src.utils import read_config

class Classifer:
    def __init__(self):
        config = read_config()
        model_checkpoint = config['model']['save_dir']
        self.token_classifier = pipeline(
            "token-classification", model=model_checkpoint, aggregation_strategy="simple"
        )

    def _tabluar_result(self, res):
        output_text = ""
        words = [ent['word'].center(10) for ent in res]
        groups = [ent['entity_group'].center(10) for ent in res]
        scores = [(str(round(ent['score']*100,1))+"%").center(10) for ent in res]
        output_text += "words   :  " + "|".join(words) + "\n"
        output_text += "groups  :  " + "|".join(groups) + "\n"
        output_text += "scores  :  " + "|".join(scores) + "\n"
        return output_text

    def __call__(self, input_text):
        res = self.token_classifier(input_text)
        return self._tabluar_result(res)

if __name__=="__main__":
    input_text = ""
    classifier = Classifer()
    while 1:
        input_text = input("please write a text:\n\t")
        if input_text=="q":
            break
        print(f"the result is :")
        print(classifier(input_text))
