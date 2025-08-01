import evaluate
from datasets import load_dataset 
import tensorflow_hub as hub 
from sentence_transformers import SentenceTransformer 
import tensorflow as tf 
import math 
import pandas as pd 
import numpy as np
from scipy.stats import pearsonr, spearmanr  # 방법 2용

# 방법 1: evaluate 라이브러리 사용
stsb_metric = evaluate.load('glue', 'stsb')

# 방법 2: 수동으로 correlation 계산하는 함수 (백업용)
def compute_sts_metrics(predictions, references):
    pearson_corr, _ = pearsonr(predictions, references)
    spearman_corr, _ = spearmanr(predictions, references)
    return {
        'pearson': pearson_corr,
        'spearmanr': spearman_corr,
        'combined_score': (pearson_corr + spearman_corr) / 2
    } 
stsb = load_dataset('glue', 'stsb') 

### USE 모델
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") 

### 베이스라인 모델
distilroberta = SentenceTransformer('stsb-distilroberta-base-v2') 

### 코사인 유사도
def use_sts_benchmark(batch): 
  # datasets 컬럼을 리스트로 변환
  sentence1_list = list(batch['sentence1'])
  sentence2_list = list(batch['sentence2'])
  
  sts_encode1 = tf.nn.l2_normalize(use_model(tf.constant(sentence1_list)),axis=1) 
  sts_encode2 = tf.nn.l2_normalize(use_model(tf.constant(sentence2_list)),axis=1) 
  cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1,sts_encode2),axis=1) 
  clip_cosine_similarities = tf.clip_by_value(cosine_similarities,-1.0,1.0)
  scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi 
  return scores 

### RoBERTa 모델 수정
def roberta_sts_benchmark(batch): 
  # datasets 컬럼을 리스트로 변환하여 sentence_transformers와 호환성 문제 해결
  sentence1_list = list(batch['sentence1'])
  sentence2_list = list(batch['sentence2'])
  
  sts_encode1 = tf.nn.l2_normalize(tf.constant(distilroberta.encode(sentence1_list)),axis=1) 
  sts_encode2 = tf.nn.l2_normalize(tf.constant(distilroberta.encode(sentence2_list)),axis=1) 
  cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1,sts_encode2),axis=1) 
  clip_cosine_similarities = tf.clip_by_value(cosine_similarities,-1.0,1.0) 
  scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi 
  return scores 

use_results = use_sts_benchmark(stsb['validation']) 
distilroberta_results = roberta_sts_benchmark(stsb['validation']) 

references = [item['label'] for item in stsb['validation']]

# TensorFlow 텐서를 numpy로 변환
use_results_np = use_results.numpy() if hasattr(use_results, 'numpy') else use_results
distilroberta_results_np = distilroberta_results.numpy() if hasattr(distilroberta_results, 'numpy') else distilroberta_results

# 방법 1: evaluate 라이브러리 사용
results = { 
      "USE": stsb_metric.compute( 
                predictions=use_results_np, 
                references=references), 
      "DistillRoberta": stsb_metric.compute( 
                predictions=distilroberta_results_np, 
                references=references) 
}

# 방법 2: 수동 계산 (백업용)
# results = { 
#       "USE": compute_sts_metrics(use_results_np, references), 
#       "DistillRoberta": compute_sts_metrics(distilroberta_results_np, references) 
# } 

print(pd.DataFrame(results))