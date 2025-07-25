from datasets import load_dataset 
from transformers import AutoTokenizer, \
                         AutoModelForQuestionAnswering, \
                         TrainingArguments, \
                         Trainer, \
                         default_data_collator, \
                         pipeline

squad = load_dataset("squad_v2") 
print(squad)

model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)

max_length = 384 
doc_stride = 128 
example = squad["train"][173] 

tokenized_example = tokenizer( 
    example["question"], 
    example["context"], 
    max_length=max_length, 
    truncation="only_second", 
    return_overflowing_tokens=True, 
    stride=doc_stride 
)

for input_ids in tokenized_example["input_ids"][:2]: 
    print(tokenizer.decode(input_ids)) 
    print("-"*50) 

def prepare_train_features(examples, pad_on_right=True): 
    tokenized_examples = tokenizer( 
        examples["question" if pad_on_right else "context"], 
        examples["context" if pad_on_right else "question"], 
        truncation="only_second" if pad_on_right else "only_first", 
        max_length=max_length, 
        stride=doc_stride, 
        return_overflowing_tokens=True, 
        return_offsets_mapping=True, 
        padding="max_length", 
    ) 

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping") 
    offset_mapping = tokenized_examples.pop("offset_mapping") 

    tokenized_examples["start_positions"] = [] 
    tokenized_examples["end_positions"] = [] 
    for i, offsets in enumerate(offset_mapping): 
        input_ids = tokenized_examples["input_ids"][i] 
        cls_index = input_ids.index(tokenizer.cls_token_id) 
        sequence_ids = tokenized_examples.sequence_ids(i) 
        sample_index = sample_mapping[i] 
        answers = examples["answers"][sample_index] 
        if len(answers["answer_start"]) == 0: 
            tokenized_examples["start_positions"].append(cls_index) 
            tokenized_examples["end_positions"].append(cls_index) 
        else: 
            start_char = answers["answer_start"][0] 
            end_char = start_char + len(answers["text"][0]) 
            token_start_index = 0 
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0): 
                token_start_index += 1 
            token_end_index = len(input_ids) - 1 
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0): 
                token_end_index -= 1 
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char): 
                tokenized_examples["start_positions"].append(cls_index) 
                tokenized_examples["end_positions"].append(cls_index) 
            else: 
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char: 
                    token_start_index += 1 
                tokenized_examples["start_positions"].append(token_start_index - 1) 
                while offsets[token_end_index][1] >= end_char: 
                    token_end_index -= 1 
                tokenized_examples["end_positions"].append(token_end_index + 1) 
    return tokenized_examples 

tokenized_datasets = squad.map(prepare_train_features, batched=True, remove_columns=squad["train"].column_names)

model = AutoModelForQuestionAnswering.from_pretrained(model) 

args = TrainingArguments( 
    f"test-squad", 
    eval_strategy = "epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=16, 
    per_device_eval_batch_size=16, 
    num_train_epochs=3, 
    weight_decay=0.01, 
)

data_collator = default_data_collator 

trainer = Trainer( 
    model, 
    args, 
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["validation"], 
    data_collator=data_collator, 
    tokenizer=tokenizer, 
) 

trainer.train() 

trainer.save_model("distillBERT_SQUAD")

qa_model = pipeline('question-answering',
                    model='distilbert-base-cased-distilled-squad',
                    tokenizer='distilbert-base-cased') 

question = squad["validation"][0]["question"] 
context = squad["validation"][0]["context"] 
print("Question:") 
print(question) 
print("Context:") 
print(context) 

print(qa_model(question=question, context=context))