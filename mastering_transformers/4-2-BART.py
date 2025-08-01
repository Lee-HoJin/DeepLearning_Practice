from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, pipeline
import pprint

model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

nlp=pipeline("summarization",
     model=model,
     tokenizer=tokenizer)

text='''

We order two different types of jewelry from this

company the other jewelry we order is perfect.

However with this jewelry I have a few things I

don't like. The little Stone comes out of these

and customers are complaining and bringing them

back and we are having to put new jewelry in their

holes. You cannot sterilize these in an autoclave

...[truncated]'''

q=nlp(text)


pp = pprint.PrettyPrinter(indent=0, width=100)

pp.pprint(q[0]['summary_text'])