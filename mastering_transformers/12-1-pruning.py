from sentence_transformers import SentenceTransformer 
from datasets import load_dataset  # 이건 그대로
import evaluate  # 새로 추가
import math 
import tensorflow as tf 
from torch.nn.utils import prune 
import torch.nn as nn
import pandas as pd 

distilroberta = SentenceTransformer('stsb-distilroberta-base-v2')

# load_metric → evaluate.load로 변경
stsb_metric = evaluate.load('glue', 'stsb') 
stsb = load_dataset('glue', 'stsb') 

mrpc_metric = evaluate.load('glue', 'mrpc') 
mrpc = load_dataset('glue','mrpc')


def roberta_sts_benchmark(batch): 
    sts_encode1 = tf.nn.l2_normalize(distilroberta.encode(batch['sentence1']),axis=1) 
    sts_encode2 = tf.nn.l2_normalize(distilroberta.encode(batch['sentence2']),axis=1) 
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2),axis=1) 
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities,-1.0,1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi 
    return scores

references = stsb['validation'][:]['label'] 

state_dict = distilroberta.state_dict()

distilroberta_results = roberta_sts_benchmark(stsb['validation']) 


# 모델의 각 linear layer에 직접 pruning 적용
for name, module in distilroberta.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)

# pruning mask를 실제 weight에 적용
for name, module in distilroberta.named_modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')

distilroberta.load_state_dict(state_dict) 

distilroberta_results_p = roberta_sts_benchmark(stsb['validation']) 


result = pd.DataFrame({ 
  "DistillRoberta":stsb_metric.compute(predictions=distilroberta_results, references=references),
  "DistillRobertaPruned":stsb_metric.compute(predictions=distilroberta_results_p, references=references)
}) 

print(result)