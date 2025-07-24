import datasets
from transformers import DistilBertTokenizerFast

conll2003 = datasets.load_dataset("conll2003")

print(conll2003["train"][0])
print(conll2003["train"].features["ner_tags"])

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
