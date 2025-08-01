import torch
from transformers import pipeline

classifier = pipeline(
    task="text-classification",
    model="bhadresh-savani/electra-base-emotion",
    torch_dtype=torch.float16,
    device=0
)
print(classifier("This restaurant has amazing food!"))