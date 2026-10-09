## 36. 추론 배포와 서빙

학습과 평가를 마친 모델을 실제로 서비스에서 쓰려면 배포 과정을 거쳐야 한다. LoRA로 학습한 모델은 배포 방식에 선택지가 있고, 서빙 엔진과 양자화 형식에 따라 속도와 비용이 달라진다. 이 장에서는 어댑터 병합과 분리 서빙의 선택, 주요 서빙 도구, 그리고 간단한 배포 예제를 다룬다.

### 어댑터 병합 vs 분리 서빙

LoRA 학습의 결과물은 작은 어댑터 가중치이다. 이를 서비스에 올리는 방법은 두 가지이다.

어댑터 병합(merge)은 BA를 원래 가중치 W0에 더해 하나의 완전한 모델로 만드는 방식이다. 병합 후에는 일반 모델과 똑같이 취급할 수 있어 어떤 서빙 엔진에서든 바로 쓸 수 있다. 추론 시 추가 연산이 없어 속도 면에서 유리하다. 단점은 베이스 모델 전체 크기의 파일을 새로 저장해야 하므로 디스크 공간을 차지하고, 여러 어댑터를 바꿔 가며 쓰기 어렵다는 점이다.

분리 서빙은 베이스 모델은 그대로 두고 어댑터만 따로 올려 추론 시점에 결합하는 방식이다. vLLM 같은 엔진은 여러 LoRA 어댑터를 하나의 베이스 모델에 동시에 서빙할 수 있어, 어댑터별로 다른 서비스를 운영할 때 메모리를 크게 절약할 수 있다. 어댑터 교체도 파일 하나만 바꾸면 되므로 배포가 가볍다. 단점은 엔진이 LoRA 서빙을 지원해야 하고, 약간의 연산 오버헤드가 있다는 점이다.

선택 기준은 단순하다. 어댑터가 하나뿐이고 최대한 단순하게 운영하고 싶다면 병합, 여러 어댑터를 하나의 GPU에서 돌리거나 자주 교체한다면 분리 서빙을 고른다.

병합은 PEFT의 merge_and_unload 메서드로 간단히 수행한다.

```python
# merge_adapter.py — LoRA 어댑터를 베이스 모델에 병합
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-2-9b-it"
ADAPTER_PATH = "outputs/dpo-final"   # 또는 outputs/sft-final
MERGED_PATH = "outputs/merged-model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
merged = model.merge_and_unload()   # 어댑터를 가중치에 합침

merged.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("병합 완료:", MERGED_PATH)
```

### vLLM으로 서빙하기

vLLM은 LLM 서빙의 사실상 표준 엔진이다. PagedAttention 같은 최적화로 처리량이 높고, OpenAI 호환 API를 제공해 기존 클라이언트 코드를 그대로 쓸 수 있다. LoRA 어댑터도 네이티브로 지원한다.

병합된 모델을 vLLM으로 서빙하는 방법은 다음과 같다.

```bash
# vLLM 서버 실행 (병합 모델 기준)
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged-model \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port 8000
```

어댑터를 분리 서빙하려면 베이스 모델을 올리면서 어댑터 목록을 함께 지정한다.

```bash
# vLLM에서 LoRA 어댑터 분리 서빙
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-9b-it \
  --enable-lora \
  --lora-modules support-bot=outputs/sft-final \
  --dtype bfloat16 \
  --port 8000
```

서버가 뜨면 OpenAI 클라이언트로 호출할 수 있다.

```python
# vllm_client.py — vLLM 서버 호출 예제
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="support-bot",   # 분리 서빙 시 어댑터 이름, 병합 시 모델 경로
    messages=[
        {"role": "system", "content": "당신은 친절한 고객 상담 직원입니다."},
        {"role": "user", "content": "배송이 왜 이렇게 늦어요?"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

### Ollama와 GGUF 양자화

개인 PC나 작은 서버에서 가볍게 돌리고 싶다면 Ollama와 GGUF 형식을 고려한다. GGUF는 llama.cpp 생태계의 모델 파일 형식으로, 4비트나 8비트 양자화된 모델을 CPU와 GPU에서 효율적으로 실행할 수 있다.

병합된 모델을 GGUF로 변환하는 흐름은 다음과 같다. 먼저 llama.cpp의 변환 스크립트로 GGUF 파일을 만들고, 필요하면 양자화를 적용한다.

```bash
# GGUF 변환과 양자화 (llama.cpp 필요)
python llama.cpp/convert_hf_to_gguf.py outputs/merged-model \
  --outfile model-f16.gguf --outtype f16
llama.cpp/llama-quantize model-f16.gguf model-q4km.gguf Q4_K_M
```

변환된 GGUF 파일은 Ollama의 Modelfile로 등록해 쓸 수 있다.

```
# Modelfile
FROM ./model-q4km.gguf
PARAMETER temperature 0.7
SYSTEM "당신은 친절한 고객 상담 직원입니다."
```

```bash
ollama create support-bot -f Modelfile
ollama run support-bot "배송이 왜 이렇게 늦어요?"
```

Ollama는 설치와 사용이 간단해 프로토타입이나 사내 소규모 배포에 적합하다. 다만 대규모 트래픽 처리에는 vLLM 같은 전용 서빙 엔진이 낫다.

### GPU 사양 고르기

서빙에 필요한 GPU는 모델 크기와 동시 요청량으로 정한다. 9B 모델을 16비트로 서빙하면 가중치만 약 18GB이므로, 여유를 두면 24GB급 GPU 1장이 적당하다. 4비트 양자화 추론을 쓰면 12GB급에서도 가능하다.

동시 요청이 많으면 메모리가 더 필요하다. vLLM은 KV 캐시를 위해 추가 메모리를 쓰는데, 동시 처리량과 시퀀스 길이에 비례한다. 처음에는 여유 있게 잡고 모니터링하면서 조정한다. 트래픽이 적다면 Ollama로 작은 GPU나 CPU에서 시작해도 된다.

### 모니터링과 롤백

서비스 중인 모델은 계속 지켜봐야 한다. 기본적으로 볼 지표는 요청 수, 평균 응답 시간, 에러율이다. LLM 특유의 지표로는 답변이 비정상적으로 짧거나 길어지는 비율, 특정 거부 문구의 빈도 같은 것도 있다.

이상 징후가 보이면 바로 이전 버전으로 롤백할 수 있어야 한다. 그래서 모델 파일과 어댑터에 버전을 붙이고, 이전 버전을 보관한다. vLLM이라면 어댑터 파일만 교체해 재시작하면 되므로 롤백이 빠르다.

사용자 피드백 채널도 모니터링의 일부이다. 싫어요 버튼이나 신고 기능을 두면 모델이 놓치는 문제를 조기에 발견할 수 있다. 이런 피드백은 다음 학습의 선호도 데이터로도 활용할 수 있어 일석이조이다.

### 보안과 접근 제어

모델 서버를 외부에 열 때는 접근 제어가 필요하다. vLLM의 OpenAI 호환 서버는 기본적으로 인증이 없으므로, 앞에 리버스 프록시를 두고 API 키를 검사하는 구성을 권장한다. 사내망에서만 쓴다면 방화벽으로 접근 IP를 제한한다.

입력 길이 제한도 둔다. max-model-len을 크게 잡으면 악의적인 긴 입력으로 메모리를 고갈시킬 수 있다. 실제 사용 범위에 맞게 제한하고, 요청 속도 제한(rate limit)을 함께 둔다.

### 따라 하기: 배포 전 검증 스크립트

배포 과정을 스크립트로 정리해 두면 반복 배포가 쉬워진다.

```python
# deploy.py — 모델 배포 전 검증과 서버 기동 정보를 정리하는 스크립트
import os
import json

MODEL_PATH = "outputs/merged-model"
REQUIRED_FILES = ["config.json", "tokenizer.json", "model.safetensors"]

print("배포 전 검사")
ok = True
for name in REQUIRED_FILES:
    path = os.path.join(MODEL_PATH, name)
    exists = os.path.exists(path)
    print(("있음" if exists else "없음"), ":", name)
    ok = ok and exists

# 모델 설정에서 파라미터 규모 확인
with open(os.path.join(MODEL_PATH, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
print("모델 타입:", cfg.get("model_type"))
print("최대 위치:", cfg.get("max_position_embeddings"))

if ok:
    print("검사 통과. 다음 명령으로 서버를 실행하세요:")
    print("python -m vllm.entrypoints.openai.api_server \\")
    print("  --model", MODEL_PATH, "\\")
    print("  --dtype bfloat16 --max-model-len 4096 --port 8000")
else:
    print("필수 파일이 없습니다. 병합 과정을 다시 확인하세요.")
```

### 배포 전 체크리스트

배포하기 전에 다음을 확인한다.

- 평가를 통과한 최종 체크포인트(또는 병합 모델)인지 확인한다. 개발 중간의 체크포인트가 섞이지 않도록 파일명을 명확히 한다.
- 시스템 프롬프트를 학습 때 쓴 형식과 일치시킨다. 학습과 다른 형식을 쓰면 성능이 떨어진다.
- 동시 요청 수와 응답 시간 목표를 정하고 부하 테스트를 한다. vLLM의 동시 처리량은 GPU 메모리와 max-model-len 설정에 따라 달라진다.
- 로깅과 모니터링을 준비한다. 이상 답변이 나오면 원인을 추적할 수 있어야 한다.
- 모델 파일과 어댑터 파일의 버전을 기록한다. 문제가 생겼을 때 롤백할 수 있어야 한다.

### 핵심 정리

- 어댑터가 하나이고 단순한 운영을 원하면 병합, 여러 어댑터를 공유하거나 자주 교체하면 분리 서빙을 선택한다.
- vLLM은 높은 처리량과 OpenAI 호환 API, LoRA 네이티브 지원으로 서비스용 서빙의 기본 선택지이다.
- Ollama와 GGUF 양자화는 개인 PC나 소규모 환경에서 가볍게 모델을 돌리는 방법이다.
- 배포 전에는 최종 모델 확인, 프롬프트 형식 일치, 부하 테스트, 로깅, 버전 기록을 체크한다.
