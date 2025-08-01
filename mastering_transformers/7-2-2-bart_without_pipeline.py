from transformers import AutoModelForSequenceClassification, AutoTokenizer 

nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli") 
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli") 

premise = "one day I will see the world" 
label = "travel" 
hypothesis = f'This example is {label}.' 

# 간단한 수정: truncation_strategy를 truncation으로 변경
x = tokenizer(
    premise, 
    hypothesis, 
    return_tensors='pt', 
    truncation=True  # 'only_first'는 기본값이므로 생략 가능
)

# 모델에 전체 입력 전달 (더 안전한 방법)
logits = nli_model(**x).logits

# 또는 원래 방식대로 input_ids만 사용하려면:
# logits = nli_model(x['input_ids']).logits

entail_contradiction_logits = logits[:,[0,2]] 
probs = entail_contradiction_logits.softmax(dim=1) 
prob_label_is_true = probs[:,1] 
print(f"Probability that '{premise}' is about {label}: {prob_label_is_true.item():.4f}")

# 원래 출력 형식을 원한다면:
# print(prob_label_is_true)