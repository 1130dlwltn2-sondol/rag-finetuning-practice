## 34. DPO와 선호도 최적화

SFT(지도 파인튜닝)로 모델이 원하는 형식과 스타일을 따라 하게 만들었다면, 다음 단계는 사람의 선호에 맞게 정렬(alignment)하는 것이다. 같은 질문에 두 가지 답이 있을 때 어느 쪽이 더 좋은지를 가르치는 과정이다. 예전에는 RLHF(사람 피드백 기반 강화학습)가 이 역할을 맡았지만, 복잡한 강화학습 파이프라인 대신 쓸 수 있는 DPO가 등장하면서 선호도 최적화가 훨씬 쉬워졌다.

### SFT 후 정렬 단계

SFT와 선호도 최적화는 목적이 다르다. SFT는 "이렇게 답하라"는 모범 답안을 보여 주는 학습이다. 반면 선호도 최적화는 "이 답이 저 답보다 낫다"는 상대적 판단을 가르친다. SFT만 거친 모델은 형식을 잘 따르지만, 미묘한 품질 차이(더 도움이 되는 답, 더 정직한 답)를 구분하는 능력은 부족할 수 있다.

일반적인 순서는 SFT를 먼저 수행한 뒤 그 결과 모델에 선호도 최적화를 적용하는 것이다. SFT 없이 바로 선호도 학습을 하면 모델이 기본적인 지시 수행 능력부터 흔들릴 수 있기 때문이다. 데이터도 두 종류가 필요하다. SFT용 모범 답안 데이터와, 선호도용 chosen·rejected 쌍 데이터이다.

### chosen과 rejected 선호도 데이터

선호도 데이터는 하나의 프롬프트에 두 개의 답변을 붙인 형태이다. 사람이 더 선호하는 답변을 chosen, 그렇지 않은 답변을 rejected라고 한다.

```json
{
  "prompt": "파이썬에서 리스트의 중복을 제거하는 방법을 알려줘.",
  "chosen": "set으로 변환한 뒤 다시 리스트로 만들면 됩니다. 예를 들어 list(set(my_list)) 와 같이 작성합니다. 단, 순서가 보장되지 않으니 순서가 중요하면 dict.fromkeys(my_list)를 사용하세요.",
  "rejected": "set을 쓰세요."
}
```

좋은 선호도 데이터를 만드는 요령은 chosen과 rejected의 차이가 학습 목표와 직접 연결되도록 하는 것이다. 위 예제에서는 단순한 답과 설명이 충분한 답의 차이를 가르친다. 두 답이 모두 틀리거나, 차이가 무작위하면 학습 효과가 떨어진다.

데이터 수집 방법은 여러 가지이다. 사람이 직접 순위를 매기는 것이 가장 정확하지만 비용이 든다. 실무에서는 강한 모델(예: 대형 상용 모델)로 후보 답변들을 생성한 뒤 사람이 검수하는 방식을 많이 쓴다. 기존 SFT 모델이 만든 답변과 사람이 고친 답변을 쌍으로 묶는 것도 좋은 방법이다.

### DPO의 개념

DPO(Direct Preference Optimization)는 선호도 데이터를 직접 이용해 모델을 최적화하는 방법이다. 기존 RLHF는 보상 모델을 따로 학습시킨 뒤 강화학습(PPO)으로 정책을 업데이트하는 복잡한 3단계 파이프라인이었다. DPO는 보상 모델을 명시적으로 만들지 않고, 선호도 확률을 직접 최대화하는 손실 함수 하나로 같은 효과를 낸다.

개념적으로 DPO의 손실은 이렇게 동작한다. 모델이 chosen 답변에 매기는 확률은 올리고, rejected 답변에 매기는 확률은 내린다. 이때 기준이 되는 참조 모델(보통 SFT를 마친 모델)로부터 너무 멀어지지 않도록 제약을 건다. 이 제약을 조절하는 하이퍼파라미터가 beta이다. beta가 크면 참조 모델에 가깝게 유지되고, 작으면 선호도 데이터에 강하게 맞춘다. 보통 0.1 정도를 시작값으로 쓴다.

DPO의 장점은 구현이 단순하고 학습이 안정적이라는 점이다. 강화학습 특유의 불안정성(보상 해킹, 학습 발산)이 거의 없어 소규모 팀에서도 다루기 쉽다. 단점으로는 선호도 데이터의 품질에 민감하고, 분포 밖의 질문에는 효과가 제한적이라는 점이 있다.

### DPO의 변형들

DPO가 나온 뒤 여러 변형이 제안되었다. ORPO는 참조 모델 없이 단일 단계로 SFT와 선호도 최적화를 함께 수행한다. 파이프라인이 단순해지지만, SFT와 DPO를 분리했을 때보다 세밀한 제어가 어렵다.

KTO는 chosen과 rejected 쌍이 아니라, 각 답변이 좋은지 나쁜지에 대한 이진 피드백만으로 학습한다. 쌍을 만들기 어려운 실제 서비스 로그(좋아요/싫어요 같은)에 적합하다.

SimPO는 참조 모델을 없애고 길이 편향을 보정하는 등 DPO를 단순화한 방법이다. 연구 단계의 방법들이므로, 실무에서는 DPO를 먼저 쓰고 필요할 때 변형을 검토하는 순서를 권장한다.

### 선호도 데이터는 얼마나 필요한가

선호도 데이터 규모에 대한 정답은 없지만, 경험적인 가이드가 있다. 수백 개만으로도 SFT 모델의 말투와 선호 방향을 다듬는 데는 충분한 경우가 많다. 수천 개가 있으면 더 안정적인 개선을 기대할 수 있다.

중요한 것은 양보다 쌍의 품질이다. chosen과 rejected의 차이가 명확하고 일관된 기준을 따르면 적은 데이터로도 효과가 난다. 반대로 기준이 들쭉날쭉하면 데이터를 늘려도 학습이 헤맨다. 소량으로 먼저 학습해 보고 방향이 맞는지 확인한 뒤 규모를 늘리는 것이 좋다.

### 보상 로그 읽는 법

DPOTrainer의 로그에는 rewards/chosen과 rewards/rejected 같은 값이 나온다. 학습이 잘 되면 chosen 보상은 올라가고 rejected 보상은 내려가며, 두 값의 차이(margin)가 벌어진다.

주의할 점은 보상의 절대값이 아니라 차이의 추이다. 두 보상이 함께 올라가도 차이가 벌어지면 정상이다. 차이가 좁혀지거나 역전되면 데이터에 문제가 있거나 학습률이 너무 높을 수 있다. 로그를 보면서 margin이 단조롭게 증가하는지 확인한다.

### SFT와 DPO를 한 번에: 파이프라인 정리

지금까지의 흐름을 정리하면 다음과 같다.

1. SFT 데이터로 SFTTrainer 학습을 수행한다(33장).
2. SFT 모델의 답변 품질을 held-out 테스트로 평가한다(35장).
3. 선호도 데이터(prompt, chosen, rejected)를 만든다.
4. DPOTrainer로 선호도 최적화를 수행한다.
5. 다시 평가해 DPO 전후를 비교한다.

DPO 후에도 아쉬운 점이 있으면 선호도 데이터를 보강해 한 번 더 돌릴 수 있다. 다만 반복할수록 참조 모델에서 멀어지므로, 2회 이상 반복할 때는 beta를 올리거나 평가를 꼼꼼히 한다.

### DPO 데이터 검수 체크리스트

선호도 데이터를 학습에 넣기 전에 다음을 확인한다.

- chosen 답변이 실제로 더 좋은가. 두 답이 비슷하면 학습 신호가 약해진다.
- 차이가 학습 목표와 연결되는가. 목표가 정중함인데 chosen이 더 길기만 하다면 엉뚱한 것을 가르친다.
- rejected가 너무 형편없지는 않은가. 차이가 극단적이면 모델이 자명한 것만 배운다. 적절한 난이도의 차이가 좋다.
- 프롬프트가 다양하게 분포하는가. 한 유형에 치우치면 그 유형에만 과적합된다.

검수는 사람이 표본을 읽는 것이 가장 확실하다. 전체를 다 볼 수 없으면 무작위로 10%를 뽑아 확인한다. 문제가 보이면 생성 프롬프트나 수집 기준을 고친다.

### DPO와 SFT 데이터의 관계

SFT 데이터와 선호도 데이터는 서로 보완한다. SFT 데이터는 모범 답안으로 "이렇게 하라"를 가르치고, 선호도 데이터는 "이쪽이 더 낫다"로 다듬는다. 두 데이터의 스타일이 크게 다르면 모델이 혼란스러워한다.

예를 들어 SFT에서는 간결한 답을 가르쳤는데 선호도 데이터의 chosen이 장황하면, DPO 단계에서 SFT의 효과가 희석된다. 두 단계의 데이터는 같은 톤과 형식 기준을 공유하는 것이 좋다. 데이터를 만들 때 스타일 가이드를 문서로 정리해 두면 일관성을 유지하기 쉽다.

### DPO 실패 패턴

DPO가 기대만큼 효과가 없을 때 흔한 원인을 정리한다.

첫째, 선호도 데이터의 품질 문제이다. chosen과 rejected의 차이가 모호하거나 일관성이 없으면 모델이 배울 것이 없다. 데이터를 다시 검수한다.

둘째, SFT가 부족한 상태에서 DPO를 돌린 경우이다. 기본적인 지시 수행이 안 되는 모델에 선호도를 가르쳐 봐야 효과가 제한적이다. SFT를 먼저 충분히 한다.

셋째, beta나 학습률 설정 문제이다. beta가 너무 크면 참조 모델에서 거의 움직이지 않고, 너무 작으면 이상한 방향으로 튄다. 0.1에서 시작해 0.05와 0.2를 시도해 본다.

넷째, 평가 방법이 DPO의 효과를 못 잡는 경우이다. 선호도 개선은 미묘해서 정확도 같은 거친 지표에는 안 잡힐 수 있다. 사람 평가나 LLM-as-judge로 다시 본다.

### 따라 하기: TRL DPOTrainer 예제

TRL의 DPOTrainer는 DPO 학습을 위한 클래스이다. 데이터는 prompt, chosen, rejected 세 필드로 준비한다.

```python
# train_dpo.py — DPOTrainer로 선호도 최적화 실행
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

SFT_ADAPTER = "outputs/sft-final"   # 33장에서 학습한 SFT 어댑터
BASE_MODEL = "google/gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 정책 모델: SFT 어댑터를 얹은 모델에 DPO용 LoRA를 추가로 적용
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
from peft import PeftModel
policy = PeftModel.from_pretrained(base, SFT_ADAPTER)
policy = get_peft_model(
    policy,
    LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)

dataset = load_dataset("json", data_files={
    "train": "preference_train.jsonl",   # prompt, chosen, rejected 필드
    "validation": "preference_valid.jsonl",
})

args = DPOConfig(
    output_dir="outputs/dpo",
    num_train_epochs=2,             # DPO는 보통 SFT보다 짧게
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,            # SFT보다 낮은 학습률 권장
    beta=0.1,                      # 참조 모델과의 거리 제약
    warmup_ratio=0.05,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)

trainer = DPOTrainer(
    model=policy,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("outputs/dpo-final")
```

DPO 학습률은 SFT보다 낮게 잡는 것이 일반적이다. 이미 SFT로 잘 조정된 모델을 미세하게 다듬는 단계이므로, 5e-5 정도가 무난한 출발점이다. 학습 중에는 chosen 답변에 대한 보상(reward_chosen)과 rejected에 대한 보상(reward_rejected) 로그를 확인한다. 두 값의 차이가 벌어지면 학습이 의도대로 진행되는 신호이다.

### 핵심 정리

- 선호도 최적화는 SFT 다음 단계로, 두 답변 중 어느 쪽이 더 좋은지 가르쳐 모델을 사람의 선호에 정렬한다.
- 선호도 데이터는 prompt, chosen, rejected 세 필드로 구성하며, 두 답변의 차이가 학습 목표와 연결되어야 한다.
- DPO는 보상 모델과 강화학습 없이 선호도를 직접 최적화하는 방법으로, 구현이 단순하고 학습이 안정적이다.
- beta는 참조 모델로부터의 이탈을 조절하고, 학습률은 SFT보다 낮게(5e-5 내외), 에폭은 짧게(1~2) 설정하는 것이 일반적이다.
