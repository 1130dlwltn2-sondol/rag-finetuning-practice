## 14. RAG 프레임워크: LangChain과 LlamaIndex

앞 장에서 RAG를 직접 코드로 만들었다. 검색, 프롬프트 구성, 생성을 직접 연결하니 구조가 손에 잡혔을 것이다. 그런데 실제 프로젝트에서는 문서 로더, 청킹, 벡터 DB 연동, 프롬프트 템플릿, LLM 호출을 매번 직접 짜는 것이 부담스럽다. 이 반복 작업을 대신해 주는 도구가 RAG 프레임워크다. 이 장에서는 대표적인 두 프레임워크인 LangChain과 LlamaIndex를 비교하고, 언제 무엇을 쓰면 좋은지 정리한다.

### 프레임워크가 필요한 이유

RAG를 구성하는 부품은 생각보다 많다. PDF나 웹 페이지에서 텍스트를 추출하는 문서 로더, 문서를 자르는 텍스트 분할기, 임베딩 모델 연동, 벡터 데이터베이스 연동, 프롬프트 템플릿 관리, LLM API 호출, 대화 기록 관리까지. 이것들을 직접 다 짜면 코드가 금방 수백 줄이 된다.

프레임워크는 이 부품들을 표준화된 인터페이스로 제공한다. 임베딩 모델을 OpenAI에서 BGE-m3로 바꾸고 싶으면 설정 한 줄만 바꾸면 된다. 벡터 DB를 Chroma에서 Qdrant로 바꾸는 것도 마찬가지다. 부품 교체가 쉬우니 실험 속도가 빨라지고, 실수로 인한 버그도 줄어든다.

다만 프레임워크는 마법이 아니다. 내부에서 무슨 일이 일어나는지 모르고 쓰면 문제가 생겼을 때 디버깅이 어렵다. 앞 장에서 RAG를 직접 만들어 본 경험이 있어야 프레임워크를 제대로 쓸 수 있다. 뼈대를 이해한 뒤 도구를 쓰는 순서가 맞다.

### LangChain의 특징

LangChain은 LLM 애플리케이션 전반을 위한 범용 프레임워크다. RAG는 그 기능 중 하나일 뿐, 에이전트, 도구 호출, 대화 메모리, 워크플로우 구성까지 폭넓게 다룬다.

강점은 유연성과 생태계다. 지원하는 모델, 벡터 DB, 문서 로더의 종류가 가장 많다. 새로운 모델이나 서비스가 나오면 LangChain 연동이 가장 먼저 나오는 경우가 많다. RAG에서 시작해 나중에 에이전트나 복잡한 워크플로우로 확장할 계획이라면 LangChain이 자연스러운 선택이다.

약점은 추상화가 복잡하다는 점이다. 체인, 러너블, 에이전트 같은 개념이 많아 처음에는 학습 곡선이 가파르다. 버전 업데이트가 빠르다 보니 코드가 자주 바뀌고, 예전 예제가 최신 버전에서 안 돌아가는 경우도 있다. 버전을 고정하고 공식 문서의 마이그레이션 가이드를 따라가는 습관이 필요하다.

### LlamaIndex의 특징

LlamaIndex는 데이터 검색과 RAG에 특화된 프레임워크다. "LLM을 위한 데이터 프레임워크"를 표방하며, 문서 인덱싱과 검색 품질에 집중한다.

강점은 검색 중심의 편의성이다. 문서를 읽어서 인덱스로 만드는 과정이 매우 간단하고, 계층적 인덱싱, 문서 요약 기반 검색 같은 고급 검색 기법이 내장되어 있다. 검색 품질을 빠르게 끌어올리고 싶은 경우, 특히 문서 구조가 복잡한 경우에 잘 맞는다.

약점은 범용성이 LangChain보다 좁다는 점이다. RAG 바깥의 복잡한 에이전트 워크플로우를 만들 때는 LangChain 쪽 생태계가 더 크다. 프로젝트가 검색 중심에서 벗어나 복잡해지면 한계를 느낄 수 있다.

### 언제 무엇을 쓸까

| 상황 | 권장 |
|---|---|
| RAG 프로토타입을 빠르게 만들고 싶다 | 둘 다 가능. 검색 품질 실험이 중심이면 LlamaIndex |
| 나중에 에이전트, 도구 호출로 확장할 계획이다 | LangChain |
| 문서 구조가 복잡하고 검색 품질을 정교하게 다뤄야 한다 | LlamaIndex |
| 지원하는 모델, DB 종류가 최대한 많아야 한다 | LangChain |
| 프레임워크 없이 직접 짠 코드로 충분하다 | 둘 다 쓰지 않는다 |

마지막 행이 중요하다. 간단한 사내 Q&A처럼 요구사항이 명확하고 바뀌지 않는다면, 앞 장처럼 직접 짠 코드가 가장 투명하고 유지보수가 쉽다. 프레임워크는 복잡도가 늘어날 때 도입하는 것이 순서다.

### 따라 하기 실습: LangChain으로 RAG 체인 만들기

pip install langchain langchain-community langchain-openai chromadb를 설치한다. OpenAI API 키가 필요한 예제이므로, 키가 없다면 앞 장의 직접 구현 방식을 먼저 연습하자.

```python
# langchain_rag.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "여기에_키_입력"

def build_chain(data_dir="data/raw", persist_dir="data/chroma_lc"):
    # 1. 문서 로드와 분할
    docs = []
    for filename in sorted(os.listdir(data_dir)):
        docs.extend(TextLoader(os.path.join(data_dir, filename), encoding="utf-8").load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 2. 임베딩과 벡터 DB 저장
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 프롬프트와 체인 구성
    prompt = ChatPromptTemplate.from_template(
        "아래 문서를 근거로 질문에 답하세요. "
        "문서에 없는 내용은 모른다고 답하세요.\n\n"
        "[문서]\n{context}\n\n[질문]\n{question}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n".join(d.text for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()
    answer = chain.invoke("입사 6개월차 직원의 연차는 어떻게 되나요?")
    print(answer)
```

앞 장의 직접 구현과 비교해 보자. 문서 로드, 분할, 임베딩, 저장, 검색, 프롬프트, 호출이 각각 한 줄씩으로 줄었다. 대신 retriever나 RunnablePassthrough 같은 개념을 알아야 읽을 수 있다. 이 트레이드오프가 프레임워크 사용의 실체다.

### 따라 하기 실습: LlamaIndex로 인덱스 만들기

pip install llama-index llama-index-llms-openai llama-index-embeddings-openai를 설치한다.

```python
# llamaindex_rag.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "여기에_키_입력"

if __name__ == "__main__":
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding()

    # 디렉터리의 문서를 읽어 인덱스를 만든다 (청킹과 임베딩이 자동으로 처리됨)
    documents = SimpleDirectoryReader("data/raw").load_data()
    index = VectorStoreIndex.from_documents(documents)

    # 쿼리 엔진으로 질문한다
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("입사 6개월차 직원의 연차는 어떻게 되나요?")
    print(response)
```

LlamaIndex는 인덱스 생성이 특히 간결하다. from_documents 한 줄로 청킹과 임베딩이 처리된다. 검색 품질과 관련된 고급 기능(리랭킹, 하이브리드 검색 등)도 쿼리 엔진 설정으로 붙이기 쉬운 구조다.

### 프레임워크를 쓸 때의 주의점

첫째, 버전을 고정한다. requirements.txt에 버전을 명시하고, 업그레이드할 때는 변경 로그를 확인한다. 특히 LangChain은 메이저 업데이트에서 API가 바뀌는 일이 잦다.

둘째, 추상화 뒤를 들여다본다. 검색 결과가 이상하면 retriever가 실제로 어떤 조각을 가져왔는지 출력해 본다. 프롬프트가 어떻게 조립되는지 확인한다. 앞 장의 직접 구현을 이해하고 있다면 이 디버깅이 어렵지 않다.

셋째, 필요 이상으로 복잡하게 만들지 않는다. 체인에 체인을 엮고 에이전트를 붙이다 보면 무엇이 문제인지 알 수 없는 시스템이 된다. 단순한 RAG로 요구사항을 만족할 수 있다면 단순하게 유지한다.

### LangChain의 핵심 개념 짧게

LangChain 코드를 읽다 보면 낯선 개념이 몇 개 나온다. 최소한으로 알아두자.

LCEL(LangChain Expression Language)은 부품을 파이프(|)로 엮는 문법이다. 앞 장 예제의 chain = ({"context": ..., "question": ...} | prompt | llm | StrOutputParser()) 부분이 LCEL이다. 왼쪽의 출력이 오른쪽의 입력으로 들어가는 구조라 데이터 흐름이 눈에 잘 들어온다.

Retriever는 "질문을 넣으면 문서 목록을 돌려주는" 표준 인터페이스다. 벡터 DB가 Chroma든 Qdrant든, as_retriever()로 감싸면 같은 방식으로 쓸 수 있다. 검색 부품을 교체할 때 체인 코드를 고칠 필요가 없는 이유다.

RunnablePassthrough는 입력을 그대로 다음 단계로 넘기는 부품이다. 예제에서는 사용자의 질문을 프롬프트의 {question} 자리에 그대로 넣는 역할을 했다. 작은 부품이지만 체인을 조립할 때 자주 쓰인다.

### 두 프레임워크를 함께 쓰는 경우

LangChain과 LlamaIndex는 경쟁 관계이기도 하지만 함께 쓰기도 한다. 흔한 조합은 LlamaIndex로 검색 파트를 만들고, LangChain으로 그 바깥의 워크플로우를 감싸는 것이다. 검색 품질은 LlamaIndex의 인덱싱 기능으로 챙기고, 에이전트나 복잡한 분기는 LangChain으로 처리한다.

다만 두 개를 함께 쓰면 의존성이 늘고 디버깅이 복잡해진다. 프로젝트 초반에는 하나로 시작해서, 실제로 한계에 부딪혔을 때만 다른 하나를 들이는 것이 좋다. "나중에 필요할 것 같아서" 미리 두 개를 다 쓰는 것은 유지보수 부담만 늘린다.

프레임워크를 고를 때 마지막으로 확인할 것은 문서의 최신성이다. 두 프레임워크 모두 업데이트가 빠르므로, 블로그 글이나 1년 전 예제를 그대로 복사하면 안 돌아가는 경우가 많다. 공식 문서의 최신 예제를 기준으로 삼고, 버전은 requirements에 고정한다.

### 핵심 정리

- 프레임워크는 문서 로더, 청킹, 임베딩, 벡터 DB 연동, 프롬프트 관리를 표준화된 부품으로 제공해 RAG 개발 속도를 높인다.
- LangChain은 범용 LLM 애플리케이션 프레임워크로 생태계가 크고 확장성이 좋지만 학습 곡선이 가파르다.
- LlamaIndex는 검색과 RAG에 특화되어 인덱싱이 간결하고 검색 품질 기능이 풍부하지만 범용 워크플로우에는 약하다.
- 요구사항이 단순하면 프레임워크 없이 직접 구현하는 것이 가장 투명하며, 프레임워크는 복잡도가 필요할 때 도입한다.
- 버전을 고정하고, 추상화 뒤의 실제 동작(검색 결과, 프롬프트)을 확인하는 습관이 디버깅의 핵심이다.
