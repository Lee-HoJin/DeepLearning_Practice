from transformers import pipeline 
import pandas as pd 

classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli") 
sequence_to_classify = "one day I will see the world" 
candidate_labels = ['travel', 
                    'cooking', 
                    'dancing', 
                    'exploration'] 

result = classifier(sequence_to_classify, candidate_labels) 
print("exclusive__")
print(pd.DataFrame(result))

result = classifier(sequence_to_classify,  
                      candidate_labels,  
                      multi_label=True) 
print("multi-label = True__")
print(pd.DataFrame(result))