import os
import pandas as pd

# imdb_df = pd.read_csv("IMDB Dataset.csv")
# reviews = imdb_df.review.to_string(index = None)

# with open("corpus.txt", "w") as f :
#     f.writelines(reviews)
    
from tokenizers import BertWordPieceTokenizer
# bert_wordpiece_tokenizer = BertWordPieceTokenizer()
# bert_wordpiece_tokenizer.train("corpus.txt")

# os.makedirs("tokenizer", exist_ok=True)
# bert_wordpiece_tokenizer.save_model("tokenizer")

# tokenizer = BertWordPieceTokenizer.from_file("tokenizer/vocab.txt")

# print(tokenizer.encode("Oh it works just fine").tokens)

# print(tokenizer.encode("ohoh i thought it might be working well").tokens)

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(tokenizer = tokenizer,
                                file_path="corpus.txt",
                                block_size=128)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = True,
    mlm_probability = 0.15
)

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir = "BERT",
    overwrite_output_dir = True,
    num_train_epochs = 1,
    per_device_train_batch_size = 128
)

from transformers import BertConfig, BertForMaskedLM
bert = BertForMaskedLM(BertConfig())

from transformers import Trainer
trainer = Trainer(
    model = bert,
    args = training_args,
    data_collator = data_collator,
    train_dataset = dataset
)

trainer.train()
trainer.save_model("MyBERT")