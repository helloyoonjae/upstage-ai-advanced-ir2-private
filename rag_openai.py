# https://github.com/openai/openai-cookbook/blob/db3144982aa26b87a9bdfb692b4fbedfdf8a14d5/examples/Get_embeddings_from_dataset.ipynb
# https://github.com/openai/openai-python/blob/release-v0.28.0/openai/embeddings_utils.py
# https://github.com/openai/openai-cookbook/blob/main/examples/Semantic_text_search_using_embeddings.ipynb
# https://github.com/openai/openai-python/pull/404
# FROM python:3.8

# RUN pip install git+https://github.com/harupy/openai-python.git@add-init.py
# RUN python -c 'from openai.api_resources.embedding import Embedding'
# https://community.openai.com/t/where-is-the-original-openai-embeddings-utils-in-the-latest-version/479854/3


import os
import json
import openai
from elasticsearch import Elasticsearch, helpers

import pickle

def load_embeddings_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

# Example usage
embeddings_file_path = "/data/ephemeral/home/data/embeddings_text-embedding-3-large_dim2048.pkl"
documents_embeddings = load_embeddings_from_pickle(embeddings_file_path)


def load_documents_from_jsonl(file_path):
    docs = []
    with open(file_path, 'r') as file:
        for line in file:
            doc = json.loads(line.strip())
            docs.append(doc)
    return docs

# Load documents
docs_file_path = '/data/ephemeral/home/data/documents.jsonl'
documents = load_documents_from_jsonl(docs_file_path)





from elasticsearch import Elasticsearch

es_username = "elastic"
es_password = "9eEvpAOlIys1-A+jPY-+"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'],
                   basic_auth=(es_username, es_password),
                   ca_certs='/data/ephemeral/home/elasticsearch-8.8.0/config/certs/http_ca.crt')


# Elasticsearch client 정보 확인
print(es.info())


from elasticsearch import Elasticsearch, helpers

# Assuming 'es' is your Elasticsearch client instance
def index_documents_with_embeddings(index_name, documents):
    actions = [
        {
            '_index': index_name,
            '_id': documents[idx]['docid'],  # Correct indexing here
            '_source': {
                'content': documents[idx]['content'],  # Correct indexing here
                'embeddings': documents_embeddings[idx].tolist()  # Correctly mapping embedding with document index
            }
        }
        for idx in range(len(documents))
    ]
    print("index 생성 완료")
    helpers.bulk(es, actions)


# Load your documents and embeddings
index_name = 'test'

# Index the documents with embeddings
index_documents_with_embeddings(index_name, documents)







  
    
    
    
    

# 역색인을 이용한 검색
def sparse_retrieve(query_str, query_embedding, size):
    body = {# MAP 0.8045 MRR 0.8061
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query_str,
                            "fields": ["content^2"],
                            "boost": 0.0025,
                            "fuzziness": "AUTO" # 오타나 철자가 약간 다른 단어들도 검색 결과에 포함
                        }
                    }
                ],
                "minimum_should_match": 1 # should 최소 하나 이상의 조건을 만족하는 문서
            }
        },
        "knn": {
            "field": "embeddings",
            "query_vector": query_embedding.tolist(),
            "k": 7,
            "num_candidates": 70,
            "boost": 1.5
        },
        "size": size
    }
    
    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test",body=body)



def search_retrieve(query_str, query_embedding, documents_embeddings, documents, n=3):
    # Step 1: Perform sparse retrieval to get initial results
    sparse_result = sparse_retrieve(query_str, query_embedding, 5)
    #print("sparse_result: ", sparse_result)
    
    # Step 2: Extract document IDs from sparse retrieval results
    doc_ids = [rst["_id"] for rst in sparse_result['hits']['hits'] if "_id" in rst]
    print("doc_ids: ", doc_ids)

    # Step 3 & 4: Retrieve the embeddings and document contents for the retrieved document IDs
    retrieved_embeddings = []
    doc_contents = []
    valid_doc_ids = []

    for doc_id in doc_ids:
        matching_doc_index = next((i for i, doc in enumerate(documents) if doc['docid'] == doc_id), None)
        if matching_doc_index is not None:
            valid_doc_ids.append(doc_id)
            doc_contents.append(documents[matching_doc_index]['content'])
            retrieved_embeddings.append(documents_embeddings[matching_doc_index])

    # Check if retrieved_embeddings is empty to avoid further errors
    if not retrieved_embeddings:
        print("No embeddings found for the retrieved document IDs.")
        return None

    # Step 5: Calculate cosine distance between query embedding and retrieved document embeddings
    distances = distances_from_embeddings(query_embedding, retrieved_embeddings, distance_metric="L2")

    # Convert distances to similarity scores (lower distance means higher similarity)
    similarities = [1 - distance for distance in distances]

    print(len(valid_doc_ids), len(doc_contents), len(similarities))

    # Step 6: Create a DataFrame for easier sorting and selection
    results_df = pd.DataFrame({
        "docid": valid_doc_ids,
        "content": doc_contents,
        "similarity": similarities
    })

    # Step 7: Sort by similarity in descending order and return top n results
    top_results = results_df.sort_values(by="similarity", ascending=False).head(n)

    return top_results










# 색인을 위한 setting 설정
settings = {
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
    "index": {
            "similarity": {
                "lm_jelinek_mercer": { 
                    "type": "LMJelinekMercer", 
                    "lambda": 0.7
                } 
            }
        },
    "analysis": {
        "analyzer": {
            "nori_index_analyzer": { 
                "type": "custom",
                "tokenizer": "nori_tokenizer",# Elasticsearch에서 제공하는 한국어 형태소 분석기로, 텍스트를 의미 있는 단어 단위로 분리합니다.
                "decompound_mode": "mixed", # 복합어를 어떻게 처리할지 결정합니다. 'mixed' 설정은 원래 형태와 분리된 형태 모두를 토큰으로 생성합니다.
                "user_dictionary":"/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/user_dictionary.txt",
                "filter": ["nori_posfilter"]                 
            },
            "nori_search_analyzer": { 
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "filter": ["nori_posfilter", 
                           "my_synonym_filter1", 
                           # "my_synonym_filter2", 에러나서 뻄
                           "my_synonym_filter3", 
                           "my_synonym_filter4",
                           "my_synonym_filter5",
                           "my_synonym_filter6",
                           "my_synonym_filter7"]  # 검색 시점에서 동의어 필터 사용
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",# 특정 품사를 제외시키는 필터로, 여기서는 어미, 조사, 구분자 등을 제외하도록 설정되어 있습니다.
                # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언, 보조 명사, 접속사, 감탄사 등 "VCP", "VX","NNB","MAJ","IC",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN" ]
            },
            "my_synonym_filter1": {
                "type": "synonym",
                "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms1.txt",
            },
            # "my_synonym_filter2": {
            #     "type": "synonym",
            #     "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms2.txt",
            # },
            "my_synonym_filter3": {
                "type": "synonym",
                "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms3.txt",
            },
            "my_synonym_filter4": {
                "type": "synonym",
                "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms4.txt",
            },
            "my_synonym_filter5": {
                "type": "synonym",
                "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms5.txt",
            },
             "my_synonym_filter6": {
                "type": "synonym",
                "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms6.txt",
            },
              "my_synonym_filter7": {
                "type": "synonym",
                "synonyms_path": "/data/ephemeral/home/elasticsearch-8.8.0/config/synonyms/synonyms7.txt",
            }
            
        }
    }
}

# 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
# 색인의 구조를 정의하며, 각 필드가 어떻게 색인되고 저장될지를 지정합니다.
mappings = {
    "properties": {
        "content": {
            "type": "text",
            "analyzer": "nori_index_analyzer",       # 색인 시점 분석기
            "search_analyzer": "nori_search_analyzer" # 검색 시점 분석기
        },
        "embeddings": {
            "type": "dense_vector",
            #"dims": 768,
            "dims": len(documents_embeddings[0]),
            "index": True,
            "similarity": "l2_norm"
        }
    }
}






#export OPENAI_API_KEY="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"

from openai import OpenAI
import traceback

client = OpenAI(api_key="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")


import numpy as np

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


#from utils.embeddings_utils import get_embedding, cosine_similarity --> openai에서 삭제함

    
def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   cut_dim = client.embeddings.create(input=[text], model=model).data[0].embedding[:2048]
   norm_dim = normalize_l2(cut_dim)
   #print(f"Generated embedding for text: {text[:30]}... -> {norm_dim[:5]}...")  # 출력문 추가
   return norm_dim

import pandas as pd
from typing import List
from scipy import spatial

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[float]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    #print(f"Distances calculated: {distances[:5]}")  # 출력문 추가
    return distances

def texts_to_tensor(texts, model="text-embedding-3-large", pprint=True):
    print(f"Input texts: {texts[:3]}")  # 입력 텍스트 디버깅

    # Generate embeddings for each text in the list
    embeddings = [get_embedding(text, model) for text in texts]
    #print(f"Generated embeddings for all texts. Sample: {embeddings[:2]}")  # 임베딩 디버깅

    # Use the first embedding as the query embedding
    query_embedding = embeddings[0]
    #print(f"Query embedding: {query_embedding[:5]}")  # 쿼리 임베딩 디버깅

    return query_embedding








# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        
        #embeddings = get_embedding(contents)
        embeddings = texts_to_tensor(contents)
        
        batch_embeddings.extend(embeddings)
        #print(f'batch {i}')
    return batch_embeddings

# 재 검색
import pandas as pd
import traceback

# 기존 CSV 파일을 사용하여 재검색 수행
def rescore_existing_results(input_filename, output_filename):
    try:
        # CSV 파일에서 데이터 읽기
        test_query = pd.read_json(input_filename, lines=True)
        
        # 검색 결과를 저장할 리스트 초기화
        updated_results = []

       # 각 row의 standalone_query 컬럼에 대해 검색 수행
        for idx, row in test_query.iterrows():
            standalone_query = row.get('standalone_query')

            if pd.notna(standalone_query) and isinstance(standalone_query, str) and len(standalone_query.strip()) > 0:
                
                # 수정된 부분에서 사용
                query_embedding = texts_to_tensor([standalone_query], pprint=False)
                search_result = search_retrieve(standalone_query, query_embedding, documents_embeddings, documents)
                
                response = {"eval_id": row.get("eval_id"), "standalone_query": standalone_query, "topk": [], "references": []}
                if search_result is not None:
                    for _, row in search_result.iterrows():
                        response["topk"].append(row["docid"])
                        response["references"].append({"similarity": row["similarity"], "content": row["content"]})
                else:
                    response = {"eval_id": row.get("eval_id"), "standalone_query": standalone_query, "topk": [], "references": []}

                updated_results.append(response)
            else:
                response = {"eval_id": row.get("eval_id"), "standalone_query": standalone_query, "topk": [], "references": []}
                updated_results.append(response)
                
        # 결과를 새로운 JSONL 파일로 저장
        with open(output_filename, "w") as of:
            for result in updated_results:
                of.write(f'{json.dumps(result, ensure_ascii=False)}\n')

    except Exception as e:
        traceback.print_exc()

# 기존 CSV 파일을 사용하여 재검색 결과 생성
rescore_existing_results("/data/ephemeral/home/sample_submission6_roberta_sparse_fullprompt.csv", "/data/ephemeral/home/sample_submission11_openai_hybrid.csv")