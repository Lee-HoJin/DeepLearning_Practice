import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from transformers import pipeline

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "The cat is so sad ."

encoded_output = tokenizer(text, return_tensors='pt')
output = model(**encoded_output)

fillmask= pipeline("fill-mask", model="roberta-base",
                   tokenizer = tokenizer)
print(pd.DataFrame(fillmask("The cat is so <mask> .")))