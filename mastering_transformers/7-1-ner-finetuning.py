import datasets
from transformers import DistilBertTokenizerFast

def main() :

    conll2003 = datasets.load_dataset("conll2003")

    print(conll2003["train"][0])
    print(conll2003["train"].features["ner_tags"])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize_and_align_labels(examples) :
        tokenized_inputs = tokenizer(examples["tokens"],
                                     truncation = True,
                                     is_split_into_words = True)
        
        labels = []

        for i, label in enumerate(examples["ner_tags"]) :
            word_ids = tokenized_inputs.word_ids(batch_index = i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids :
                if word_idx is None :
                    label_ids.append(-100)

                elif word_idx != previous_word_idx :
                    label_ids.append(label[word_idx])

                else :
                    label_ids.append(label[word_idx] if label_all_tokens else -100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels

        return tokenized_inputs
    

    print(tokenize_and_align_labels(conll2003['train'][4:5]))

    return 0