## 13. 첫 RAG 시스템 만들기

지금까지 배운 조각을 하나로 엮어 본다. 이 장에서는 문서 준비부터 청킹, 임베딩, 저장, 검색, 생성을 거쳐 답을 내는 RAG 시스템 전체를 직접 만들어 본다. 단계마다 코드가 이어지도록 구성했으니 순서대로 실행해 보자.

### 전체 흐름 다시 보기

RAG 시스템은 두 개의 흐름으로 나뉜다. 인덱싱 흐름은 미리 실행되는 준비 단계고, 질의 흐름은 사용자가 질문할 때마다 실행되는 단계다.

인덱싱 흐름: 원본 문서를 모은다 → 적절한 크기로 자른다(청킹) → 각 조각을 임베딩 벡터로 바꾼다 → 벡터 데이터베이스에 저장한다.

질의 흐름: 질문이 들어온다 → 질문을 임베딩 벡터로 바꾼다 → 벡터 데이터베이스에서 가까운 조각을 검색한다 → 조각들과 질문을 프롬프트에 묶는다 → LLM이 답을 생성한다.

인덱싱은 문서가 바뀔 때만 다시 하면 된다. 질의는 요청마다 반복된다. 이 분리를 머릿속에 넣어 두면 나중에 성능을 개선할 때도 어디를 손대야 할지 감이 온다.

### 1단계: 문서 준비

실습용 문서를 준비한다. 실제 서비스에서는 PDF, 사내 위키, 고객 상담 기록 등에서 가져오겠지만, 여기서는 텍스트 파일 몇 개로 대체한다.

```python
# prepare_docs.py
import os

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

sample_docs = {
    "leave_policy.txt": """2026년 개정 휴가 규정
제1조(연차 부여): 입사 1년 미만 직원도 월 1일씩 연차를 사용할 수 있다.
제2조(연차 수당): 미사용 연차에 대해서는 통상임금의 100%를 수당으로 지급한다.
제3조(반차): 오전 반차와 오후 반차를 사용할 수 있으며, 연차 0.5일씩 차감한다.""",
    "remote_work.txt": """재택근무 지침
제1조(대상): 전 직원은 주 2일까지 재택근무를 신청할 수 있다.
제2조(승인): 재택근무는 팀장의 사전 승인이 필요하며, 주 단위로 신청한다.
제3조(장비): 재택근무 시 필요한 장비는 IT팀에 요청할 수 있다.""",
    "meeting_room.txt": """회의실 이용 안내
제1조(예약): 회의실 예약은 사내 인트라넷에서 최대 2주 전까지 가능하다.
제2조(취소): 예약 취소는 사용 1시간 전까지 가능하며, 무단 불참은 3회 누적 시 예약 권한이 제한된다.""",
}

for filename, text in sample_docs.items():
    with open(os.path.join(RAW_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text)
print(f"{len(sample_docs)}개 문서를 {RAW_DIR}에 저장했다.")
```

### 2단계: 청킹

문서를 통째로 임베딩하면 검색 품질이 떨어진다. 한 문서 안에 여러 주제가 섞여 있으면 벡터가 그 중간 어딘가에 위치해 질문과 정확히 매칭되기 어렵다. 그래서 문서를 주제 단위로 자른다. 이를 청킹(chunking)이라고 한다.

청킹의 기본 규칙은 세 가지다. 첫째, 한 조각은 하나의 주제를 담는다. 위 예제처럼 조항 단위로 자르는 것이 이상적이다. 둘째, 조각의 크기는 적당히 유지한다. 너무 짧으면 문맥이 부족하고, 너무 길면 특정 주제가 희석된다. 셋째, 조각마다 출처 정보를 메타데이터로 남긴다. 어느 문서의 몇 번째 조각인지 알면 나중에 답의 근거를 제시할 수 있다.

```python
# chunking.py
import os

RAW_DIR = "data/raw"

def chunk_text(text, source):
    # 빈 줄을 기준으로 조각을 나누고 출처를 메타데이터로 붙인다
    chunks = []
    for i, part in enumerate(text.split("\n\n")):
        part = part.strip()
        if not part:
            continue
        chunks.append({
            "id": f"{source}-{i}",
            "text": part,
            "metadata": {"source": source, "chunk_index": i},
        })
    return chunks

if __name__ == "__main__":
    all_chunks = []
    for filename in sorted(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, filename)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, filename))
    print(f"총 {len(all_chunks)}개 조각으로 나뉘었다.")
    for c in all_chunks[:3]:
        print(f"[{c['id']}] {c['text'][:40]}...")
```

청킹 전략은 검색 품질에 큰 영향을 준다. 고정 크기, 문장 경계, 의미 기반, 계층 구조 등 다양한 방법이 있는데, 자세한 내용은 Part 2의 청킹 장에서 다룬다. 여기서는 조항 단위 분할로 시작한다.

### 3단계: 임베딩과 저장

조각들을 벡터 데이터베이스에 넣는다. Chroma를 쓰면 임베딩과 저장이 한 번에 처리된다.

```python
# indexing.py
import chromadb

CHUNKS = [
    # chunking.py의 결과를 이어받는다고 가정한다
]

def build_index(chunks, persist_dir="data/chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection("company_docs")
    except Exception:
        pass
    collection = client.create_collection(name="company_docs")
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"{len(chunks)}개 조각을 인덱스에 저장했다.")
    return collection

if __name__ == "__main__":
    # 실제로는 chunking.py의 all_chunks를 import해서 쓴다
    from chunking import chunk_text
    import os
    all_chunks = []
    for filename in sorted(os.listdir("data/raw")):
        with open(os.path.join("data/raw", filename), encoding="utf-8") as f:
            all_chunks.extend(chunk_text(f.read(), filename))
    build_index(all_chunks)
```

### 4단계: 검색

질문이 오면 벡터 데이터베이스에서 관련 조각을 찾는다. 상위 몇 개를 가져올지는 설계 선택이다. 너무 적으면 근거가 부족하고, 너무 많으면 프롬프트가 길어지고 노이즈가 섞인다. 처음에는 3~5개로 시작하는 것이 무난하다.

```python
# retrieve.py
import chromadb

def retrieve(query, n_results=3, persist_dir="data/chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection("company_docs")
    result = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]

if __name__ == "__main__":
    hits = retrieve("입사 6개월차인데 연차를 며칠 쓸 수 있나요?")
    for h in hits:
        print(f"거리 {h['distance']:.3f} | {h['metadata']['source']} | {h['text'][:50]}...")
```

### 5단계: 생성

검색된 조각들을 프롬프트에 넣어 LLM에 전달한다. 생성 단계의 프롬프트 설계가 답변의 신뢰도를 좌우한다. 핵심은 세 가지 지시다. 문서를 근거로 답할 것, 문서에 없는 내용은 모른다고 답할 것, 근거가 된 문서의 출처를 함께 밝힐 것.

```python
# generate.py

def build_rag_prompt(query, retrieved):
    context_lines = []
    for i, hit in enumerate(retrieved, start=1):
        src = hit["metadata"]["source"]
        context_lines.append(f"[문서{i} | 출처: {src}]\n{hit['text']}")
    context = "\n\n".join(context_lines)
    return f"""당신은 사내 규정을 안내하는 도우미입니다.
아래 [참고 문서]의 내용만을 근거로 질문에 답하세요.
참고 문서에 없는 내용은 답하지 말고 "해당 내용은 문서에서 찾을 수 없습니다"라고 답하세요.
답변 마지막에는 근거가 된 문서의 출처를 함께 적으세요.

[참고 문서]
{context}

[질문]
{query}

[답변]
"""

def call_llm(prompt):
    # 실제 서비스에서는 여기서 LLM API를 호출한다
    # 예: OpenAI API, 사내 LLM 게이트웨이, Ollama 등
    raise NotImplementedError("LLM 호출 코드를 여기에 연결하세요.")

if __name__ == "__main__":
    from retrieve import retrieve
    question = "입사 6개월차인데 연차를 며칠 쓸 수 있나요?"
    hits = retrieve(question)
    prompt = build_rag_prompt(question, hits)
    print(prompt)
```

### 전체 실행 순서

위 스크립트들을 순서대로 실행하면 첫 RAG 시스템이 완성된다.

1. python prepare_docs.py — 샘플 문서 생성
2. python indexing.py — 청킹, 임베딩, 저장
3. python generate.py — 질문 검색과 프롬프트 생성 확인
4. call_llm에 실제 LLM 호출을 연결하면 답변이 생성된다

이 구조가 RAG의 최소 단위다. 실무에서는 청킹을 정교하게 하고, 검색에 하이브리드 검색과 리랭킹을 붙이고, 프롬프트를 다듬는 식으로 이 뼈대에 살을 붙여 나간다. 그 과정이 Part 2의 주제다.

### 핵심 정리

- RAG 시스템은 인덱싱 흐름(문서 준비, 청킹, 임베딩, 저장)과 질의 흐름(검색, 프롬프트 구성, 생성)으로 나뉜다.
- 청킹은 한 조각에 하나의 주제를 담는 것이 원칙이며, 출처 메타데이터를 함께 남겨 답의 근거를 추적할 수 있게 한다.
- 검색 결과 수는 3~5개로 시작하고, 프롬프트에는 근거 문서만으로 답하고 모르는 것은 모른다고 말하라는 지시를 넣는다.
- 이 장의 구조가 RAG의 최소 단위이며, 검색 품질 개선은 Part 2에서 이 뼈대 위에 추가한다.
