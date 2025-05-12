import numpy as np
import random
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')

sent = '근육이 커지기 위해서는'

input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
print(input_ids)

output = model.generate(input_ids,
                        max_length=128,
                        repetition_penalty=2.0,
                        use_cache=True)
output_ids = output.numpy().tolist()[0]
print(output_ids)

tokenizer.decode(output_ids)

output = model(input_ids)
top5 = tf.math.top_k(output.logits[0, -1], k=5)

print(tokenizer.convert_ids_to_tokens(top5.indices.numpy()))

sent = '근육이 커지기 위해서는'
input_ids = tokenizer.encode(sent)

while len(input_ids) < 50:
    output = model(np.array([input_ids]))
    # Top 5의 단어들을 추출
    top5 = tf.math.top_k(output.logits[0, -1], k=5)
    # Top 5의 단어들 중 랜덤으로 다음 단어로 선택.
    token_id = random.choice(top5.indices.numpy())
    input_ids.append(token_id)

print(tokenizer.decode(input_ids))
