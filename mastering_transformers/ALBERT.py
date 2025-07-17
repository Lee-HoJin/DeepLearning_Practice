import pandas as pd
from transformers import AlbertTokenizer, AlbertModel
from transformers import pipeline

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
model = AlbertModel.from_pretrained("albert-base-v2")

text = 'The cat is so sad .'

encoded_output = tokenizer(text, return_tensors = 'pt')
output = model(**encoded_output)

fillmask = pipeline('fill-mask',
                    model = 'albert-base-v2')
print(pd.DataFrame(fillmask("The cat is so [MASK] ")))

