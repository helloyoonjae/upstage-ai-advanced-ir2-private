import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)

model = SentenceTransformer("snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS")
#model = SentenceTransformer("jhgan/ko-sroberta-multitask")
#model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#model = SentenceTransformer("/data/ephemeral/home/code/output/training_jhgan-ko-sroberta-multitask_binary_2024-10-09_16-45-53-epoch-1")
#model = SentenceTransformer("/data/ephemeral/home/code/sroberta_finetuned_epoch_1")
#model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# {"eval_id": 32, "msg": [{"role": "user", "content": "오늘 너무 즐거웠다!"}]} --> 과학상식으로 standalone_query를 뽑음
# --> 이 문제는 embedding문제가 아니라 standalone_query를 이상하게 뽑은 llm문제

# 임베딩 생성
sentences = ["This is an example sentence."]
embeddings = model.encode(sentences)

# 임베딩 차원 확인
print("임베딩 차원:", embeddings.shape[1])

# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        combined_texts = [
            f"{doc['title']} {doc['content']} {', '.join(doc['keywords'])} {doc['summary']}"
            for doc in batch
        ]     
        embeddings = get_embedding(combined_texts)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings


# 새로운 index 생성
import time

def create_es_index(index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)
    # 5초 동안 대기
    print("인덱스가 생성되었습니다. 3초 동안 대기합니다...")
    time.sleep(3)
    print("대기 완료.")
    
    es.indices.refresh(index=index)
    # 5초 동안 대기
    print("인덱스가 리프레시되었습니다. 3초 동안 대기합니다...")
    time.sleep(3)
    print("대기 완료.")
    


# 지정된 인덱스 삭제
def delete_es_index(index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ElasticsearchWarning

def bulk_add(index, docs):
    actions = [
        {
            '_index': index,
            '_source': doc
        } for doc in docs
    ]

    try:
        # 대량 인덱싱을 한 번만 시도하고 결과를 반환합니다.
        response = helpers.bulk(es, actions)
        print("Successfully indexed {} documents.".format(response[0]))
        return response
    except ElasticsearchWarning as e:
        # 실패한 문서와 에러 메시지를 출력하고 함수를 종료합니다.
        print("Error indexing documents:")
        print(e.errors)
    except Exception as e:
        # 다른 예외도 적절히 처리합니다.
        print("An unexpected error occurred:", str(e))



# 역색인을 이용한 검색
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str#,
                #"boost": 2.0  # 쿼리에 가중치를 부여하여 중요도를 높임
            }
        }
    }
    
    return es.search(index="test", 
                     query=query, 
                     size=size, 
                     #sort="_score:desc",  # 점수에 따라 내림차순 정렬
                     track_total_hits=True,  # 정확한 검색 결과 총 개수 추적
                     explain=True,
                     request_cache=False                 
    )




# # Vector 유사도를 이용한 검색
def dense_retrieve(query_str, size, sparsedocids=None):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]
    #query_embedding /= np.linalg.norm(query_embedding)  # Normalize the query vector: 정답이 이상함

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn_query = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 1000
    }
    return es.search(index="test", knn=knn_query, size=size)

    
def dense2_retrieve(query_str, size, sparsedocids=None):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]
    #query_embedding /= np.linalg.norm(query_embedding)  # Normalize the query vector: 정답이 이상함 
    
    knn_query = {
        "script_score": { # 각 문서의 유사도를 점수로 매깁니다
            "query": {
                "match_all": {} # 모든 문서 검색
            },
            "script": {
                "source": "1 / (1 + l2norm(params.query_vector, 'embeddings'))",
                # "1 / (1 + cosineSimilarity(params.query_vector, 'embeddings'))",
                "params": {
                    "query_vector": query_embedding.tolist()
                }
            }
        }
    }        

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", query=knn_query, size=size)
   




# 역색인 + Vector 유사도 혼합
def hybrid_retrieve(query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    
    body = {# MAP 0.8045 MRR 0.8061
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query_str,
                            "fields": ["content^3", "keywords^5", "title^5"], 
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
            "boost": 3.5
        },
        "size": size
    }
    
    
    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test",
                     body=body,
                     allow_partial_search_results=True,
                     #sort="_score:desc",  # 점수에 따라 내림차순 정렬
                     #explain=True,
                     #highlight=True,
                     human=True,
                     track_scores=True
                     )
    
# 하이브리드 검색 함수 수정
def hybrid_docid_retrieve(query_str, size, sparsedocids=None):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str])[0]

    if sparsedocids:
        # body = {
        #     "query": {
        #         "bool": {
        #             "must": [
        #                 {
        #                     "terms": {
        #                         "_id": sparsedocids
        #                     }
        #                 }
        #             ],
        #             "should": [
        #                 {
        #                     "multi_match": {
        #                         "query": query_str,
        #                         "fields": ["content"],
        #                         "boost": 0.005
        #                     }
        #                 }
        #             ],
        #             "minimum_should_match": 1
        #         }
        #     },
        #     "knn": {
        #         "field": "embeddings",
        #         "query_vector": query_embedding.tolist(),
        #         "k": size,
        #         "num_candidates": 20,
        #         "boost": 2
        #     }
        # }
        
        if sparsedocids:
            query = {
                "script_score": { # 각 문서의 유사도를 점수로 매깁니다
                    "query": {
                        "ids": {  # 'ids' 쿼리를 사용하여 특정 문서 ID 집합을 대상으로 설정
                            "values": sparsedocids
                        }
                    },
                    "script": {
                        "source": "1 / (1 + cosineSimilarity(params.query_vector, 'embeddings'))",
                        "params": {
                            "query_vector": query_embedding.tolist()
                        }
                    }
                }
            }
       

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", query=query, size=size)

    


from elasticsearch import Elasticsearch

es_username = "elastic"
es_password = "Et-Bt310qxHNm_U9aPa*"

# Elasticsearch client 생성
# Elasticsearch client creation with error handling
try:
    es = Elasticsearch(
        ['https://localhost:9200'],
        basic_auth=(es_username, es_password),
        ca_certs='/data/ephemeral/home/elasticsearch-8.8.0/config/certs/http_ca.crt'
    )
    # Check if connection is successful
    print(es.info())
except Exception as e:
    print(f"Failed to connect to Elasticsearch: {e}")  


# # 노드 정보 조회
# nodes_info = es.cat.nodes(format="json")

# # 노드 정보 출력
# print(nodes_info)

# 색인을 위한 setting 설정
settings = {
    #https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
    "index": {
            "similarity": {
                "lm_jelinek_mercer": { 
                    "type": "LMJelinekMercer", 
                    "lambda": 0.7
                } 
            }
        },
    # "index": {
    #         "similarity": {
    #             "lm_dirichlet": { 
    #                 "type": "LMDirichlet", 
    #                 "mu": 2000
    #             } 
    #         }
    #     },
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
                           #"my_synonym_filter2", # 에러나서 뻄
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
            #"dims": 384,
            "dims": len(embeddings[0]),
            "index": True,
            "similarity": "l2_norm"#"cosine"
        }
    }
}



# # settings, mappings 설정된 내용으로 'test' 인덱스 생성
# create_es_index("test", settings, mappings)

# # 문서의 content 필드에 대한 임베딩 생성
# index_docs = []
# # with open("../data/documents.jsonl") as f:
# with open("/data/ephemeral/home/data/documents_meta_keyword.jsonl") as f:
#     docs = [json.loads(line) for line in f]
# embeddings = get_embeddings_in_batches(docs)
                
# # 생성한 임베딩을 색인할 필드로 추가
# for doc, embedding in zip(docs, embeddings):
#     doc["embeddings"] = embedding.tolist()
#     index_docs.append(doc)

# # 'test' 인덱스에 대량 문서 추가
# ret = bulk_add("test", index_docs)

# # 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
# print('색인된 총 문서: ',len(ret),'\n\n\n')




# 재 검색
import pandas as pd
import traceback

#export OPENAI_API_KEY="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"
   
from openai import OpenAI
import traceback

client = OpenAI()
llm_model = "gpt-4o"

    
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import heapq
import torch
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import numpy as np

# exp_normalize function
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

reranker_tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
reranker_model = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker")

# rescore_existing_results function
def rescore_existing_results(input_filename, output_filename):
    try:
        # Read data from the input JSONL file
        test_query = pd.read_json(input_filename, lines=True)
        print(f"Loaded {len(test_query)} queries from input file.")
        
        # Initialize a list to store the updated search results
        updated_results = []

        # Process each query in the dataframe
        for idx, row in tqdm(test_query.iterrows(), total=len(test_query), desc="Processing queries"):
            standalone_query = row.get('standalone_query')
            combined_content = row.get('combined_content')
            if pd.notna(standalone_query) and isinstance(standalone_query, str) and len(standalone_query.strip()) > 0:
                # 1-1 two way combined
                # Perform the hybrid retrieval (example function call)
                search_result_sparse = sparse_retrieve(standalone_query, 100)
                search_result_hybrid = hybrid_retrieve(standalone_query, 100)
                search_result_dense = dense_retrieve(standalone_query, 50)
                print(f"Retrieved {len(search_result_sparse['hits']['hits'])} documents for query '{standalone_query}'.")
                print(f"Retrieved {len(search_result_hybrid['hits']['hits'])} documents for query '{standalone_query}'.")
                print(f"Retrieved {len(search_result_dense['hits']['hits'])} documents for query '{standalone_query}'.")
                
                # Combine hybrid and dense results
                combined_hits = search_result_hybrid['hits']['hits'] + search_result_dense['hits']['hits']+ search_result_sparse['hits']['hits']
                # 중복된 docid 제거
                unique_docs = {hit["_source"]["docid"]: hit for hit in combined_hits}.values()
                print(f"Retrieved {len(unique_docs)} documents for query '{standalone_query}'.")
                
                # Store the top results along with their references
                docs = [
                    {"docid": hit.get("_source").get("docid"), 
                     "title": hit.get("_source").get("title"),
                     "keywords": hit.get("_source").get("keywords"),
                     "content": hit.get("_source").get("content"),
                     "summary": hit.get("_source").get("summary")
                     }
                    for hit in unique_docs
                ]
                
                #1-1 one way 
                # search_result = hybrid_retrieve(standalone_query, 100)
                # docs = [
                #     {"docid": hit.get("_source").get("docid"), 
                #      "title": hit.get("_source").get("title"),
                #      "keywords": hit.get("_source").get("keywords"),
                #      "content": hit.get("_source").get("content"),
                #      "summary": hit.get("_source").get("summary")
                #      }
                #     for hit in search_result['hits']['hits']
                # ]
                
                
                # 2-1 Re-ranking Dongjin-kr/ko-reranker
                rerank_start_time = time.time()
                pairs = [[combined_content, doc['content']] for doc in docs]
                reranker_model.eval()

                with torch.no_grad():
                    inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
                    reranker_model.to('cuda')
                    scores = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
                    scores = exp_normalize(scores.cpu().numpy())
                    indices = [(-score, idx) for idx, score in enumerate(scores)]
                    heapq.heapify(indices)
                    
                rerank_end_time = time.time()
                print(f"Re-ranking took {rerank_end_time - rerank_start_time:.2f} seconds.")    

                #Select top 5 results for further re-ranking
                topk = []
                references = []
                for _ in range(min(20, len(indices))):
                    score, idx = heapq.heappop(indices)
                    docid = docs[idx]["docid"]
                    reference = {"score": float(-score), "content": docs[idx]['content']}
                    topk.append(docid)
                    references.append(reference)
                
                # Update response with re-ranked results
                response = {
                    "eval_id": row.get("eval_id"),
                    "standalone_query": multi_query,
                    "topk": topk,
                    "references": references
                }
                updated_results.append(response)
                

                #3-1 LLM call               
                llm_results = [] 
                topk_content = ""
                for i in range(len(response.get("topk"))):
                    topk_content += f"관련문서_{i}_ID: {response.get('topk')[i]}\n"
                    topk_content += f"관련문서_{i}_내용: {response.get('references')[i].get('content')}\n"

                system_message = {
                    "role": "system",
                    "content": """
                    당신은 문서 내용의 적합성을 평가하는 전문가입니다.
                    질문과 가장 관련도가 높은 3개 문서만 추려서, 해당 문서의 ID를 알려주세요.
                    가장 관련도가 높을수록 ID를 먼저 적어주고, 다음과 같은 형태로 알려주세요.
                    b0864744-6dfd-4240-8a93-0320aac2429f,b0864744-6dfd-4240-8a93-0320aac2428f,b0864744-6dfd-4240-8a93-0320aac2427f
                    관련된 문서들이 없는 경우 아무것도 생성하지 마세요.
                    """
                }

                # 사용자 쿼리 및 문서 메시지
                query_message = {"role": "user", "content": f"질문: {combined_content}"}
                document_message = {"role": "user", "content": f"문서 5개의 ID와 내용: {topk_content}"}

                # LLM 호출을 위한 전체 메시지 배열 생성
                full_message = [system_message, query_message, document_message]


                try:
                    qaresult = client.chat.completions.create(
                        model=llm_model,
                        messages=full_message,
                        temperature=0,
                        seed=1,
                        timeout=30
                    )
                    llm_qaresult = qaresult.choices[0].message.content.strip().split(',')
                    llm_results.extend(llm_qaresult)
                    print(f"LLM 응답: {llm_results}")
                except Exception as e:
                    traceback.print_exc()
                    llm_results.append(None)  # 실패한 경우 None을 추가
                    continue  # 현재 문서가 실패해도 다음 문서로 계속 진행

                # Update the existing response
                response['topk'] = llm_results
                response['references'] = []  # Clear references if needed

                # Append only once after updating
                updated_results.append(response)
                
                
                
            else:
                # If query is not valid, append an empty result
                response = {"eval_id": row.get("eval_id"), "standalone_query": standalone_query, "topk": [], "references": []}
                print(f"유효하지 않은 쿼리이므로 빈 결과를 추가합니다: {response}")
                updated_results.append(response)

                    
                
                
        # 결과를 새로운 JSONL 파일로 저장
        with open(output_filename, "w") as of:
            for result in updated_results:
                of.write(f'{json.dumps(result, ensure_ascii=False)}\n')
        print(f"모든 결과를 '{output_filename}' 파일에 저장했습니다.")
    except Exception as e:
        traceback.print_exc()
        
        
# 기존 CSV 파일을 사용하여 재검색 결과 생성 

rescore_existing_results('/data/ephemeral/home/data/highscore_with_combined_content.jsonl', "/data/ephemeral/home/sample_submission14_snunlpKLUE_combined300_synonyms_reranking(combined)_jimiprompt_documeta100_llmcall_rerankertwice.csv")


