## 60. 실전 프로젝트: 사내 문서 Q&A 챗봇 만들기

지금까지 배운 내용을 하나의 프로젝트로 묶어보자. 목표는 사내 문서에 답하는 Q&A 챗봇을 만드는 것이다. RAG 파이프라인이 중심이고, 필요하면 파인튜닝을 결합한다. 진행 순서는 기획, 데이터 준비, 구축, 평가, 배포의 다섯 단계다. 각 단계에서 무엇을 하고, 어떤 함정을 피해야 하는지를 순서대로 따라간다.

### 단계 1. 기획: 질문부터 시작한다

프로젝트가 실패하는 가장 흔한 이유는 문서부터 모으고 질문은 나중에 생각하기 때문이다. 검색 시스템의 품질은 결국 질문에 답을 얼마나 잘하느냐로 평가되므로, 질문 목록을 먼저 만든다.

실제 업무 상황을 떠올려보자. 회사의 휴가 규정, 복지 제도, IT 지원 절차, 보안 정책 문서를 대상으로 하는 챗봇이다. 먼저 각 부서 담당자에게 "직원들이 가장 자주 묻는 질문 10개씩"을 요청한다. 이렇게 모인 질문 50개가 프로젝트의 출발점이자 나중에 평가에 쓸 테스트셋의 씨앗이 된다.

질문을 모았다면 난이도를 분류한다. 단일 문서에서 답을 찾을 수 있는 질문, 여러 문서를 종합해야 답을 찾을 수 있는 질문, 문서에 없는 답을 요구하는 질문으로 나눈다. 예를 들어 "연차는 몇 개야"는 단일 문서, "육아휴직 후 복귀 절차와 관련 서류는"은 여러 문서 종합, "올해 휴가 정책이 바뀔 예정이야"는 문서에 없는 질문이다. 마지막 유형은 챗봇이 "문서에서 찾을 수 없다"고 답해야 하므로, 그 응답 정책도 기획 단계에서 정한다.

마지막으로 성공 기준을 숫자로 정한다. "직원 만족도 80점 이상", "답변 근거 문서 제시율 100퍼센트", "답이 없는 질문에 대한 거짓 답변 0건"처럼 측정 가능한 목표가 필요하다. 목표가 없으면 평가 단계에서 무엇을 고쳐야 할지 판단할 수 없다.

기획 단계의 산출물은 세 가지다: 질문 목록 50개, 문서에 없는 질문에 대한 응답 정책, 측정 가능한 성공 기준.

### 단계 2. 데이터 준비: 문서와 질문셋

문서를 모은다. 실제 사내 문서라면 PDF, 워드, 사내 위키, 노션 페이지 등 형식이 제각각일 것이다. 형식을 통일하는 것보다 중요한 것은 메타데이터다. 문서마다 출처(부서, 문서명, 작성일, 버전)를 기록한다. 나중에 답변에 근거를 붙일 때 이 메타데이터가 그대로 쓰인다.

문서 품질도 점검한다. 오래된 문서가 섞여 있으면 검색 결과가 꼬인다. 예를 들어 2023년 휴가 규정과 2025년 개정 규정이 함께 인덱싱되어 있으면 챗봇이 둘을 섞어 답할 수 있다. 구버전 문서는 보관용으로 분리하거나, 메타데이터에 적용 기간을 명시하고 검색 시 최신 문서를 우선하도록 한다.

질문셋은 평가용과 학습용으로 나눈다. 평가용 50문항은 기획 단계에서 모은 질문에 정답 근거 문서를 연결해 만든다. 이 정답 연결(ground truth)은 사람이 직접 확인해야 한다. 나중에 검색 품질을 재는 recall@k가 이 정답 연결에 의존하기 때문이다.

```python
# prepare_evalset.py
# 기획 단계에서 모은 질문에 정답 근거 문서를 연결해 평가용 데이터셋을 만든다.
# eval_questions.json의 각 항목에는 정답이 들어 있는 문서 ID 목록을 함께 적는다.

import json

# 사람이 직접 검토한 질문과 정답 근거 문서 ID
eval_items = [
    {
        "question": "연차 유급휴가는 입사 후 언제부터 발생하나요?",
        "gold_doc_ids": ["hr-2025-leave-policy"],  # 정답이 들어 있는 문서
        "category": "단일 문서",
    },
    {
        "question": "육아휴직 후 복귀 시 필요한 서류와 신청 절차를 알려주세요.",
        "gold_doc_ids": ["hr-2025-childcare", "hr-2025-return-process"],
        "category": "다중 문서 종합",
    },
    {
        "question": "2026년 복지 포인트 제도가 바뀔 예정인가요?",
        "gold_doc_ids": [],  # 문서에 없음 -> "모름" 응답 기대
        "category": "문서 외 질문",
    },
]

with open("eval_questions.json", "w", encoding="utf-8") as f:
    json.dump(eval_items, f, ensure_ascii=False, indent=2)

print(f"평가셋 {len(eval_items)}문항 저장 완료")
```

파인튜닝을 함께 쓸 계획이라면 인스트럭션 데이터도 준비한다. 단, 이 단계에서 전체를 만들 필요는 없다. 먼저 RAG만으로 구축하고 평가한 뒤, 말투와 형식에서 부족함이 확인되면 파인튜닝 데이터를 추가하는 순서가 30장의 최후 수단 원칙과 맞는다.

### 단계 3. 구축: 최소 동작 파이프라인부터

처음부터 하이브리드 검색과 리랭킹을 붙이지 않는다. 최소 동작 파이프라인(MVP)을 먼저 만들고, 평가로 병목을 확인한 뒤 개선한다.

MVP의 구성은 단순하다. 13장에서 만든 것과 같은 흐름이다: 문서 청킹(400~512 토큰, 오버랩 10~20퍼센트), 임베딩, Chroma에 저장, 상위 k개 검색, 프롬프트에 붙여 생성. 아래 코드는 이 프로젝트의 MVP 뼈대다.

```python
# mvp_rag.py
# 사내 문서 Q&A 챗봇의 최소 동작 파이프라인

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="company_docs")
embedder = SentenceTransformer("nlpai-lab/KoE5")

# 문서 청크를 임베딩해 저장하는 함수 (문서는 미리 청킹되어 있다고 가정)
def index_chunks(chunks):
    # chunks: [{"id": "hr-2025-leave-policy#c1", "text": "...", "meta": {...}}]
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["meta"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"{len(chunks)}개 청크 인덱싱 완료")

# 질문에 답하는 함수
def answer(question, top_k=5):
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # 검색 결과를 프롬프트에 붙인다
    context = "\n\n".join(
        f"[출처: {m['doc_title']}] {d}" for d, m in zip(docs, metas)
    )
    prompt = (
        "아래 사내 문서를 참고해 질문에 답하세요. "
        "문서에 없는 내용은 모른다고 답하세요.\n\n"
        f"문서:\n{context}\n\n질문: {question}\n답변:"
    )
    # 실제 호출은 사용하는 LLM API로 교체
    # response = llm.generate(prompt)
    # return response, metas
    return prompt, metas  # MVP 단계에서는 프롬프트 구성까지만 확인

if __name__ == "__main__":
    prompt, sources = answer("연차 유급휴가는 입사 후 언제부터 발생하나요?")
    print(prompt[:500])
    print("근거 문서:", [s["doc_title"] for s in sources])
```

핵심은 두 가지다. 첫째, "문서에 없는 내용은 모른다고 답하세요"라는 지침을 프롬프트에 넣는다. 문서 외 질문에 거짓 답변을 하지 않게 하는 가장 간단한 장치다. 둘째, 출처를 답변에 함께 붙이는 구조를 처음부터 만든다. 근거 제시는 나중에 붙이기 어렵다.

### 단계 4. 평가: 어디가 병목인지 측정한다

MVP가 돌아가면 23장의 방법으로 측정한다. 검색과 생성을 분리해서 재는 것이 원칙이다. 평가셋 50문항을 돌리면서 recall@5(정답 문서가 상위 5개 안에 들어오는 비율)를 먼저 본다. 이 수치가 낮으면 검색이 병목이므로 하이브리드 검색(BM25 병렬 + RRF)이나 리랭킹을 붙인다. 수치가 높은데 답변이 틀리다면 생성이 병목이므로 프롬프트를 고치거나 모델을 바꾼다.

측정 코드는 아래처럼 단순하게 짤 수 있다.

```python
# evaluate.py
# 평가셋으로 검색 품질을 측정한다. 검색과 생성은 분리해서 본다.

import json
from mvp_rag import answer  # MVP의 answer 함수 재사용

with open("eval_questions.json", encoding="utf-8") as f:
    eval_items = json.load(f)

hits = 0
answerable = 0
for item in eval_items:
    if not item["gold_doc_ids"]:
        continue  # 문서 외 질문은 검색 평가에서 제외
    answerable += 1
    _, sources = answer(item["question"], top_k=5)
    retrieved_ids = [s["doc_id"] for s in sources]
    # 정답 문서 중 하나라도 상위 5개 안에 있으면 성공
    if any(g in retrieved_ids for g in item["gold_doc_ids"]):
        hits += 1

recall_at_5 = hits / answerable if answerable else 0
print(f"답변 가능 문항: {answerable}개")
print(f"recall@5: {recall_at_5:.2f}")

if recall_at_5 < 0.7:
    print("조치: 검색이 병목이다. 하이브리드 검색 또는 리랭킹을 검토하라")
else:
    print("조치: 검색은 양호하다. 답변 품질을 사람 평가로 확인하라")
```

검색이 병목이라면 21장의 하이브리드 검색을 붙인다. 특히 사내 용어나 약어가 많은 환경에서는 BM25가 벡터 검색의 빈틈을 잘 메운다. 그래도 부족하면 22장의 리랭킹을 추가하는데, 22장에서 강조했듯이 정밀도가 병목으로 측정될 때만 추가한다.

답변 품질은 사람이 본다. 50문항 중 정답 문서를 잘 찾아온 문항을 골라 답변이 정확한지, 근거와 일치하는지, 말투가 일관적인지를 체크한다. 이 과정에서 말투와 형식의 문제가 반복적으로 발견되면 그때 파인튜닝을 검토한다. 예를 들어 답변이 매번 너무 길거나, 문의처 형식이 제각각이라면 인스트럭션 데이터 2천~5천 건으로 SFT를 돌리는 것이 프롬프트를 늘리는 것보다 깔끔하다.

### 단계 5. 배포: 작게 열고 크게 키운다

처음부터 전사 배포하지 않는다. 한 부서(예: HR팀)에 먼저 열어 파일럿 운영을 한다. 파일럿 기간에는 모든 질문과 답변, 사용자가 누른 피드백(도움됨/도움안됨)을 로그로 남긴다. 이 로그가 다음 개선의 재료이자, 파인튜닝 데이터의 원천이 된다.

배포 형태는 상황에 맞게 고른다. 사내 메신저(슬랙, 팀즈)에 봇으로 붙이는 것이 가장 접근성이 좋다. 문서가 외부로 나가면 안 되는 환경이라면 사내 서버에 올리고, 모델도 오픈소스로 직접 서빙한다. 36장에서 다룬 vLLM이나 Ollama가 이 선택지에 들어간다.

운영 중에는 세 가지 지표를 본다. 질문 대비 답변 성공률(거짓 답변 없이 답한 비율), 근거 문서 클릭률(사용자가 출처를 실제로 여는지), 피드백 점수다. 특히 거짓 답변은 0건 목표를 유지해야 하므로, "모름" 응답 비율이 갑자기 떨어지면 검색 품질이나 프롬프트를 의심하고 조사한다.

문서 업데이트 절차도 배포의 일부다. 새 문서가 생기면 누가, 어떤 형식으로, 얼마나 자주 인덱싱하는지를 정해두지 않으면 3개월 뒤 검색 품질이 무너진다. 문서 관리 담당자와 인덱싱 주기(예: 매주 월요일 자동 재인덱싱)를 합의하고, 구버전 문서 처리 규칙(보관 분리 또는 적용 기간 메타데이터)을 문서화한다.

### 핵심 정리

- 프로젝트는 질문 목록부터 시작한다. 질문이 평가셋이 되고, 평가셋이 개선의 기준이 된다.
- 문서의 메타데이터(출처, 작성일, 버전)는 근거 제시와 구버전 관리에 쓰이므로 처음부터 기록한다.
- 최소 동작 파이프라인을 먼저 만들고 평가로 병목을 확인한 뒤 하이브리드 검색, 리랭킹, 파인튜닝을 순서대로 붙인다.
- 검색과 생성은 분리해서 측정한다. recall@k가 낮으면 검색을, 검색은 좋은데 답이 틀리면 생성을 고친다.
- 배포는 한 부서 파일럿으로 시작하고, 로그를 남겨 다음 개선과 파인튜닝 데이터로 쓴다.
