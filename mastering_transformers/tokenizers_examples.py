from transformers import AutoModel, AutoTokenizer

tokenizerTUR = AutoTokenizer.from_pretrained( "dbmdz/bert-base-turkish-uncased")
print(f"VOC size is: {tokenizerTUR.vocab_size}")

tokenizerEN = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"VOC size: {tokenizerEN.vocab_size}")