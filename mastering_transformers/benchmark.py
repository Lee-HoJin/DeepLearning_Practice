from transformers import BertTokenizer
from transformers import BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# text = "Using Transformers is easy!"
# print(tokenizer(text))

# encoded_input = tokenizer(text, return_tensors="pt")
# print(encoded_input)

# model = BertModel.from_pretrained("BERT-base-uncased")

# output = model(**encoded_input)
# print(output)

from transformers import pipeline
