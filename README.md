[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tm6AYAOm)
# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이윤재](https://github.com/UpstageAILab)             |            [장은지](https://github.com/UpstageAILab)             |            [이재명](https://github.com/UpstageAILab)             |            [전백찬](https://github.com/UpstageAILab)             |            [최지](https://github.com/UpstageAILab)             |
|                          엘라스틱서치, 리랭크, 모델학습                            |                            콜버트, 임베딩                             |                            콜버트, 크로스인코더                            |                            청킹, 페이스                             |                            전처리, 프롬프트 엔지니어링                            |
# 과학 지식 질의 응답 시스템 구축

정보 검색 2팀 - 전지재능

---

## 0. Overview

### Environment

- **Hardware**
  - CPU: AMD 라이젠 스레드리퍼 3960X 24코어 프로세서
  - GPU: 엔비디아 지포스 RTX 3090
  - CUDA 버전: 12.2
  - OS: Ubuntu 20.04

- **Software**
  - IDE: Jupyter Notebook, Visual Studio Code
  - Python 버전: 3.x

### Requirements

- **Python 라이브러리**
  - torch==1.9.0
  - torchvision==0.10.0
  - pandas==1.3.3
  - numpy==1.21.2
  - timm==0.4.12
  - elasticsearch
  - sentence-transformers
  - faiss
  - kss
  - wandb

- **GPU 자원**
  - 데이터 처리 및 모델 학습을 위한 충분한 GPU 자원

- **데이터 증강 기법**
  - 모델의 일반화 성능을 향상시키기 위한 다양한 데이터 증강 기법 적용

- **모니터링 도구**
  - 학습 과정 모니터링 및 실험 관리 도구인 `wandb` 설치 및 설정

---

## 1. Competition Info

### Overview

- **대회명**: 과학 지식 질의 응답 시스템 구축 대회
- **목표**: 질문과 이전 대화 히스토리를 기반으로 참고할 문서를 검색 엔진에서 추출한 후, 이를 활용하여 질문에 적합한 답변을 생성하는 시스템 구축
- **평가 지표**: MAP (Mean Average Precision)
- **데이터셋**:
  - **색인 대상 문서**: 4,272개의 교육 및 과학 상식 문서
    - 오픈 Ko LLM 순위표에 들어가는 Ko-H4 데이터 중 MMLU, ARC 데이터 기반
  - **평가 데이터**: 220개의 대화형 자연어 메시지

### Timeline

- **대회 시작일**: 2024년 10월 2일 (10시)
- **최종 제출 마감일**: 2024년 10월 24일 (19시)

---

## 2. Components

### Directory

```
├── code
│   ├── jupyter_notebooks
│   │   ├── 03.Meta-Llama-3.1-8B.ipynb
│   │   ├── Model_Solar.ipynb
│   │   ├── ko-gemma-2-9B-summarize.ipynb
│   │   ├── preprocessor.ipynb
│   │   └── t5.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   ├── [패스트캠퍼스] Upstage AI Lab 3기_NLP_경진대회_발표자료_11조.pdf
│   │   └── [패스트캠퍼스] Upstage AI Lab 3기_NLP_경진대회_현황공유판_11조.xlsx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

---

## 3. Data Description

### Dataset Overview

- **문서**:
  - 총 4,272개의 다양한 과학 및 교육 관련 문서
  - 각 문서는 다음의 정보를 포함함:
    - `content`: 문서의 본문 텍스트
    - `src`: 출처 정보
  - **Meta 정보** 생성:
    - LLM(`rtzr-ko-gemma-2-9b-it`)을 활용하여 각 문서의 메타 정보 생성
    - 생성된 필드:
      - `title`: 문서의 제목
      - `keywords`: 관련 키워드 목록
      - `summary`: 문서 요약
      - `categories`: 연관 카테고리 (예: 물리학, 생물학)

- **평가 데이터**:
  - 총 220개의 대화형 자연어 메시지
  - 데이터 유형:
    - 단일 턴 질문: 200개 (90.91%)
    - 멀티 턴 대화 (3턴): 20개 (9.09%)

### EDA

- **문서 길이 분석**:
  - 각 문서의 글자 수는 최소 44자에서 최대 1,230자까지 다양함
  - 평균 글자 수: 315자
  - 중앙값: 299자
  - 표준편차: 104자
  - 문서 길이의 편차가 커서 하나의 접근 방식으로는 한계 존재
  - 의미 기반 청킹(Semantic Chunking)을 통해 문서의 의미 파악과 검색 성능 향상을 도모함

- **데이터 분포 분석**:
  - 문서들은 물리학, 화학, 생물학, 공학, 컴퓨터 과학, 일반 상식 등 다양한 분야를 포함함
  - ARC 데이터(전체의 약 50%)는 도메인 정보가 부족하여 LLM을 활용한 재분류 필요

- **Meta 정보 생성**:
  - LLM을 활용하여 문서의 제목, 키워드, 요약, 연관 카테고리 등의 메타 정보 생성
  - 검색 과정에서 추가적인 피처로 활용

- **Validation Set 생성**:
  - LLM을 활용하여 전체 문서의 대화를 생성
  - 3개의 Validation Set 생성 (230개, 460개, 1,000개)
  - 대화 유형:
    - 1턴 일상 대화: 10%
    - 1턴 질의: 30%
    - 1턴 유의어 질의: 50%
    - 3턴 질의: 10%

### Data Processing

- **청킹(Chunking) 전략**:

  - **나이브 청킹**:
    - 고정된 글자 수로 분할 (예: 200자)
    - 재귀적 청킹: 문장 단위로 분할하되, 각 청크의 크기가 200자를 넘지 않도록 분할

  - **의미 기반 청킹(Semantic Chunking)**:
    - 문장 간의 의미적 유사도를 계산하여 의미적으로 일관된 청크 생성
    - Sentence Transformer로 문장 임베딩 생성 후 코사인 유사도 계산
    - 유사도 임계값(0.7)에 따라 청크 분할
    - 최대 청크 수를 제한하기 위해 K-Means 클러스터링 적용

- **데이터 증강**:

  - 모델의 일반화 성능을 향상시키기 위해 다양한 데이터 증강 기법 적용
    - **EDA 기법**:
      - 유의어 교체(Synonym Replacement, SR)
      - 임의 단어 삽입(Random Insertion, RI)
      - 단어 순서 변경(Random Swap, RS)
      - 임의 단어 삭제(Random Deletion, RD)
    - **AEDA**:
      - 다양한 문장부호를 원문에 임의 위치에 추가
    - **역번역**:
      - 데이터를 한국어에서 영어로, 다시 영어에서 한국어로 번역
    - **외부 데이터셋 활용**:
      - 공개되어 있는 대화 말뭉치를 대회 데이터셋과 동일한 포맷으로 변경하여 사용

---

## 4. Modeling

### Model Description

- **사용된 주요 사전 학습 모델**:
  - `digit82/kobart-summarization`
  - `eenzeenee/t5-base-korean-summarization`
  - `meta-llama/Meta-Llama-3.1-8B`
  - `upstage/SOLAR-10.7B-v1.0`
  - `rtzr/ko-gemma-2-9b-it`

- **임베딩 모델**:
  - `SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")`
  - `OpenAI text-embedding-ada-002`
  - `Solar embedding`

- **모델 선택 이유**:
  - 한국어로 학습된 모델이 다국어 모델보다 우수한 성능을 보여 선택
  - LLM을 활용하여 메타 정보 생성 및 쿼리 이해 향상
  - 스파스 및 밀집 검색 방법을 모두 활용하여 검색 성능 향상

### Modeling Process

- **데이터 증강**:
  - 다양한 데이터 증강 기법을 통해 데이터셋 확장
  - 모델의 일반화 성능 향상을 도모

- **파인 튜닝**:
  - 사전 학습된 모델을 불러와 특정 태스크에 맞게 추가 학습

- **임베딩 및 검색**:
  - 다양한 임베딩 모델을 평가하여 최적의 모델 선정
  - 스파스 검색(BM25)과 밀집 벡터 검색(Dense Retrieval)을 결합한 하이브리드 검색 적용

- **Elasticsearch 인덱싱**:
  - 문서와 임베딩을 Elasticsearch에 인덱싱
  - 분석기 및 토크나이저 설정 (예: 한국어 처리를 위한 `nori` 분석기)
  - 유사도 알고리즘 및 파라미터 조정 (예: `LMJelinekMercer`와 람다 값 조정)

- **재순위화(Re-ranking)**:
  - Cross-Encoder 모델을 활용하여 상위 문서에 대한 재순위화 수행
  - `bongsoo/klue-cross-encoder-v1` 모델 사용
  - LLM을 활용하여 학습 데이터를 생성하고 모델을 파인 튜닝하여 성능 향상

- **LLM 프롬프트 엔지니어링**:
  - 대화 이력으로부터 Standalone Query를 생성하기 위한 프롬프트 개발
  - 일관된 출력 형식을 유지하기 위해 프롬프트에 출력 예시와 형식에 대한 명확한 지시 포함

- **모델 학습 및 평가**:
  - `wandb` 등의 도구를 활용하여 학습 과정 모니터링
  - MAP 및 MRR 등의 지표를 활용하여 모델 평가
  - 검증 결과를 기반으로 모델을 반복적으로 개선

---

## 5. Result

### Leader Board

- **Private 리더보드**:
  - **최종 MAP 점수**: **0.926**
  - **순위**: **1위**

- **Public 리더보드**:
  - **최종 MAP 점수**: **0.84**
  - **순위**: **3위**

### Presentation

- **발표 자료**: [Upstage AI Lab 3기 NLP 경진대회 발표자료](#) *(실제 PDF 파일 링크를 삽입해주세요)*

---

## etc

### Meeting Log

- **미팅 일정**: 매일 오전 11시
- **협업 도구**:
  - **GitHub**: 코드 공유
  - **Google 스프레드시트**: 실험 기록
  - **Google Docs**: 문서화 및 노트 공유
  - **Zoom**: 온라인 미팅
  - **Slack**: 실시간 소통

### Reference

- **Elasticsearch 문서**: [Elasticsearch Reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- **OpenAI API 문서**: [OpenAI API](https://platform.openai.com/docs/api-reference)
- **SentenceTransformers 라이브러리**: [SentenceTransformers](https://www.sbert.net/)
- **관련 연구 논문**:
  - ColBERT: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)
  - Cross-Encoder를 활용한 재순위화 모델

---

감사합니다.
