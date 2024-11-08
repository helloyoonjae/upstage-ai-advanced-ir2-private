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
# def sparse_retrieve(query_str, size):
#     query = {
#         "match": {
#             "content": {
#                 "query": query_str
#             }
#         }
#     }
#     return es.search(index="test", query=query, size=size, sort="_score")

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
    
    # query = {
    #     "bool": {
    #         "must": [
    #             {"match": {"content": query_str}}
    #         ]
    #     }
    # }
    
    # query = {
    #     "bool": {
    #         "must": [
    #             {"match_phrase": {"content": query_str}}
    #         ]
    #     }
    # }


    return es.search(index="test", 
                     query=query, 
                     size=size, 
                     sort="_score:desc",  # 점수에 따라 내림차순 정렬
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
        "num_candidates": 200
    }
    
    # knn_query = {
    #     "script_score": { # 각 문서의 유사도를 점수로 매깁니다
    #         "query": {
    #             "match_all": {} # 모든 문서 검색
    #         },
    #         "script": {
    #             "source": "1 / (1 + l2norm(params.query_vector, 'embeddings'))",
    #             "params": {
    #                 "query_vector": query_embedding.tolist()
    #             }
    #         }
    #     }
    # }        

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn_query, size=size)
    return es.search(index="test", query=knn_query, size=size)

    # # sparsedocids가 있을 경우 해당 문서들로 제한하여 검색
    # print("sparsedocids: ", sparsedocids)
    # if sparsedocids:
    #     knn_query = {
    #         "script_score": { # 각 문서의 유사도를 점수로 매깁니다
    #             "query": {
    #                 "ids": {  # 'ids' 쿼리를 사용하여 특정 문서 ID 집합을 대상으로 설정
    #                     "values": sparsedocids
    #                 }
    #             },
    #             "script": {
    #                 "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
    #                 "params": {
    #                     "query_vector": query_embedding.tolist()
    #                 }
    #             }
    #         }
    #     }
    # else:
    #     # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    #     knn_query = {
    #         "script_score": { # 각 문서의 유사도를 점수로 매깁니다
    #             "query": {
    #                 "match_all": {} # 모든 문서 검색
    #             },
    #             "script": {
    #                 "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
    #                 "params": {
    #                     "query_vector": query_embedding.tolist()
    #                 }
    #             }
    #         }
    #     }        

    




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
            "similarity": "l2_norm"
        }
    }
}


#l2_norm 역색인 검색 score: 69.965576 source: 태양계에서 개기 월식이 발생할 때, 태양, 달, 지구는 다음과 같은 순서로 배열되어 있습니다. 먼저, 지구가 태양과 달 사이에 위치하게 됩니다. 그리고 달은 지구를 사이에 두고 태양과 서로 반대편 일직선 상에 위치하게 됩니다. 이러한 배열 순서로 인해 개기 월식이 발생하게 되는 것입니다. 개기 월식은 태양이 지구에 가려지는 현상으로, 달이 태양과 지구 사이에 정확히 위치하면 발생합니다. 이러한 현상은 우리가 일상적으로 관찰할 수 있는 현상 중 하나이며, 천문학적인 현상으로도 중요한 의미를 가지고 있습니다. 
#cosine 역색인 검색 score: 68.137886 source: 태양계에서 개기 월식이 발생할 때, 태양, 달, 지구는 다음과 같은 순서로 배열되어 있습니다. 먼저, 지구가 태양과 달 사이에 위치하게 됩니다. 그리고 달은 지구를 사이에 두고 태양과 서로 반대편 일직선 상에 위치하게 됩니다. 이러한 배열 순서로 인해 개기 월식이 발생하게 되는 것입니다. 개기 월식은 태양이 지구에 가려지는 현상으로, 달이 태양과 지구 사이에 정확히 위치하면 발생합니다. 이러한 현상은 우리가 일상적으로 관찰할 수 있는 현상 중 하나이며, 천문학적인 현상으로도 중요한 의미를 가지고 있습니다. 



# settings, mappings 설정된 내용으로 'test' 인덱스 생성
create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
# with open("../data/documents.jsonl") as f:
with open("/data/ephemeral/home/data/documents_meta_keyword.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print('색인된 총 문서: ',len(ret),'\n\n\n')

# #test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
# #test_query = "개기일식, 개기월식의 차이는?"
# #test_query = "반도체란?"
# #test_query = "철이 녹이 스는 건 왜 그런거야?"
# #test_query = "철이 녹는 이유"
# test_query = "예외 처리가 필요한 경우"
#test_query = "연구자가 가져야 할 태도와 자세"


# # 역색인을 사용하는 검색 예제
# search_result_retrieve = sparse_retrieve(test_query, 10)
# #{'took': 4, 'timed_out': False, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}, 'hits': {'total': {'value': 4118, 'relation': 'eq'}, 'max_score': 20.791729,
# #print(search_result_retrieve)
# # 결과 출력 테스트
# for rst in search_result_retrieve['hits']['hits']:
#     print('역색인 검색 score:', rst['_score'], 'source:', rst['_source']["content"],'\n\n\n')


# # Vector 유사도 사용한 검색 예제
# search_result_retrieve = dense_retrieve(test_query, 5)

# # 결과 출력 테스트
# for rst in search_result_retrieve['hits']['hits']:
#     print('벡터 유사도 검색 score:', rst['_score'], 'source:', rst['_source']["content"],'\n\n\n')


    
# 아래부터는 실제 RAG를 구현하는 코드입니다.
# OpenAI API 키를 환경변수에 설정
# export OPENAI_API_KEY="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"

# from openai import OpenAI
# import traceback

# client = OpenAI()
# # 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
# #llm_model = "gpt-3.5-turbo-1106"
# llm_model = "gpt-4o-mini-2024-07-18"

# #RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
# persona_qa = """
# ## 역할: 과학 상식 전문가
# ## 지시사항:
# - 사용자의 질문에 답하기 위해 제공된 참고 자료를 활용하여 간결하게 한국어로 답변을 만드세요.
# - 주어진 참고 자료로도 답을 찾을 수 없는 경우, 정보가 부족하다고 대답하세요.
# """
# # standalone_query를 반드시 생성하도록 강조하는 것이 중요
# persona_function_calling = """
# ## 역할: 과학 상식 전문가
# ## 지시사항:
# 사용자가 과학과 관련된 질문을 할 경우, standalone_query를 생성하기 위해 반드시 search tool을 호출하십시오. 
# 이때 standalone_query는 과학 상식 전문가의 지식을 활용하여 사용자의 모든 대화를 바탕으로, 질문을 구체적이고 명확한 단어를 사용하여 변형한 최종 검색어입니다. 반드시 한국어로 작성해야 하며, 검색의 효과성을 극대화할 수 있도록 핵심 키워드를 포함하십시오.
# 사용자가 일상 대화를 할 경우, search tool을 호출하지 말고 대화 맥락에 맞춰 자연스럽게 답변하십시오.
# """

# # Function calling에 사용할 함수 정의
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search",
#             "description": "search relevant documents",
#             "parameters": {
#                 "properties": {
#                     "standalone_query": {
#                         "type": "string",
#                         "description": "Final query suitable for use in search from the user messages history."
#                     }
#                 },
#                 "required": ["standalone_query"],
#                 "type": "object"
#             }
#         }
#     },
# ]


# # LLM과 검색엔진을 활용한 RAG 구현
# def answer_question(messages):
#     # 함수 출력 초기화
#     response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

#     # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
#     # A. OpenAI 모델 호출 (기본 대화 및 검색 필요 확인)
#     msg = [{"role": "system", "content": persona_function_calling}] + messages
#     try:
#         result = client.chat.completions.create(
#             model=llm_model,
#             messages=msg,
#             tools=tools,
#             #tool_choice={"type": "function", "function": {"name": "search"}},
#             temperature=0,
#             seed=1,
#             timeout=30
#         )
#     except Exception as e:
#         traceback.print_exc()
#         return response

#     # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
#     # B. 검색 엔진 호출 및 문서 추출
#     if result.choices[0].message.tool_calls:
#         tool_call = result.choices[0].message.tool_calls[0]
#         function_args = json.loads(tool_call.function.arguments)
#         standalone_query = function_args.get("standalone_query")

#         if standalone_query:
#             search_result = sparse_retrieve(standalone_query, 3)

#             response["standalone_query"] = standalone_query
#             retrieved_context = []
#             for rst in search_result['hits']['hits']:
#                 retrieved_context.append(rst["_source"]["content"])
#                 response["topk"].append(rst["_source"]["docid"])
#                 response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})
                
                
#             # 토큰 아끼기
#             # if retrieved_context:
#             #     retrieved_context_str = "\n".join(retrieved_context)
#             #     messages.append({"role": "user", "content": f"다음은 참고 자료입니다. 이를 활용해 질문에 대해 답변해 주세요: {retrieved_context_str}"})
#             #     msg = [{"role": "system", "content": persona_qa}] + messages

#             #     try:
#             #         qaresult = client.chat.completions.create(
#             #             model=llm_model,
#             #             messages=msg,
#             #             temperature=0,
#             #             seed=1,
#             #             timeout=30
#             #         )
#             #     except Exception as e:
#             #         traceback.print_exc()
#             #         return response

#             #     response["answer"] = qaresult.choices[0].message.content    

#     # 검색이 필요하지 않은 경우 바로 답변 생성
#     # else:
#     #     response["answer"] = result.choices[0].message.content

#     return response



# persona_function_calling = """
# ## 역할: 과학 상식 전문가
# ## 지시사항:
# 사용자가 과학과 관련된 질문을 할 경우, standalone_query를 생성하기 위해 반드시 search tool을 호출하십시오. 
# 이때 standalone_query는 과학 상식 전문가의 지식을 활용하여 사용자의 질문을 더 구체적이고 명확한 단어로 변형하십시오. 
# 동의어나 관련 정답 키워드를 추가하여 검색 범위를 확장하십시오. 
# 최종 standalone_query는 반드시 한국어로 작성하며, 검색의 효과성을 극대화할 수 있도록 핵심 키워드를 포함하십시오.
# 사용자가 일상 대화를 할 경우, search tool을 호출하지 말고 대화 맥락에 맞춰 자연스럽게 답변하십시오.
# """


# persona_function_calling = """# 현재 최고 점수
# ## 역할: 과학 상식 전문가
# ## 지시사항:
# 반드시 사용자가 과학과 관련된 질문을 할 경우에만, standalone_query를 생성하기 위해 반드시 search tool을 호출하십시오. 
# 이때 standalone_query는 반드시 사용자의 질문과 관련된 동의어 및 정답 키워드를 추가하여 검색 범위를 확장하십시오. 
# 최종 standalone_query는 반드시 한국어로 작성하여야 합니다.
# 사용자가 일상 대화를 할 경우, search tool을 호출하지 말고 대화 맥락에 맞춰 자연스럽게 답변하십시오.
# """

# persona_function_calling = """ 
# 역할: 사용자의 질문에 관련된 단어를 추가하여 검색어로 변경 후 search tool 호출
# 지시사항: 사용자가 정보를 요청할 경우 standalone_query를 생성하기 위해 반드시 search tool을 호출하십시오.
# standalone_query는 반드시 사용자의 질문에 사용된 단어를 모두 포함해야 하며, 
# 질문에 사용된 단어들과 관련된 단어를 도출할 수 있는 경우 5개 이하로 추가하여 검색 범위를 확장해야 합니다.
# 사용자가 '너 뭘 잘해?' 와 같은 일상적인 대화를 할 경우, search tool을 호출하지 말고 대화 맥락에 맞춰 자연스럽게 답변하십시오.
# """

# # Function calling에 사용할 함수 정의
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search",
#             "description": "search relevant documents",
#             "parameters": {
#                 "properties": {
#                     "standalone_query": {
#                         "type": "string",
#                         "description": "Final query suitable for use in search from the user messages history."
#                     }
#                 },
#                 "required": ["standalone_query"],
#                 "type": "object"
#             }
#         }
#     },
# ]

# # LLM과 검색엔진을 활용한 RAG 구현
# def answer_question(messages):
#     # 함수 출력 초기화
#     response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

#     # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
#     # A. OpenAI 모델 호출 (기본 대화 및 검색 필요 확인)
#     msg = [{"role": "system", "content": persona_function_calling}] + messages
#     try:
#         result = client.chat.completions.create(
#             model=llm_model,
#             messages=msg,
#             tools=tools,
#             temperature=0,
#             seed=1,
#             timeout=30
#         )
#     except Exception as e:
#         traceback.print_exc()
#         return response

#     # 검색이 필요한 경우 검색 호출 후 결과를 활용하여 답변 생성
#     if result.choices[0].message.tool_calls:
#         tool_call = result.choices[0].message.tool_calls[0]
#         function_args = json.loads(tool_call.function.arguments)
#         standalone_query = function_args.get("standalone_query")
#         search_type = function_args.get("search_type")

#         if standalone_query:
#             # B. 검색 방식에 따라 검색 수행 (sparse 또는 dense)
#             search_result = hybrid_retrieve(standalone_query, 3)

#             # 결과를 response에 저장
#             response["standalone_query"] = standalone_query
#             retrieved_context = []
#             for rst in search_result['hits']['hits']:
#                 retrieved_context.append(rst["_source"]["content"])
#                 response["topk"].append(rst["_source"]["docid"])
#                 response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

#     return response




# # 함수 호출 후 response 출력
# # #test_messages = [{"role": "user", "content": "철이 녹이 스는 건 왜 그런거야?"}]
# # test_messages = [{"role": "user", "content": "예외 처리가 필요한 경우를 알려줘"}]

# # response = answer_question(test_messages)
# # print(response)

# # 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
# def eval_rag(eval_filename, output_filename):
#     with open(eval_filename) as f, open(output_filename, "w") as of:
#         idx = 0
#         for line in f:
#             j = json.loads(line)
#             print(f'Test {idx}\nQuestion: {j["msg"]}')
#             response = answer_question(j["msg"])
#             print(f'Answer: {response["answer"]}\n')

#             # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용F
#             output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
#             of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
#             idx += 1

# # 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
# eval_rag("/data/ephemeral/home/data/eval.jsonl", "sample_submission9_roberta_hybrid_4omini.csv", )




# 재 검색
import pandas as pd
import traceback

# 아래부터는 실제 RAG를 구현하는 코드입니다.
# OpenAI API 키를 환경변수에 설정
# export OPENAI_API_KEY="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"

# from openai import OpenAI
# import traceback

# client = OpenAI()
# # 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
# #llm_model = "gpt-3.5-turbo-1106"
# llm_model = "gpt-4o"


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import heapq
import torch
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import numpy as np

reranker_tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
reranker_model = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker")

# exp_normalize function
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

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

            if pd.notna(standalone_query) and isinstance(standalone_query, str) and len(standalone_query.strip()) > 0:
                # 1-1 two way combined
                # Perform the hybrid retrieval (example function call)
                search_result_hybrid = hybrid_retrieve(standalone_query, 100)
                search_result_dense = dense_retrieve(standalone_query, 50)
                print(f"Retrieved {len(search_result_hybrid['hits']['hits'])} documents for query '{standalone_query}'.")
                print(f"Retrieved {len(search_result_dense['hits']['hits'])} documents for query '{standalone_query}'.")
                
                # Combine hybrid and dense results
                combined_hits = search_result_hybrid['hits']['hits'] + search_result_dense['hits']['hits']
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
                
                
                # 2-1 Re-ranking
                rerank_start_time = time.time()
                pairs = [[standalone_query, doc['content']] for doc in docs]
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

                # Select top 3 results
                topk = []
                references = []
                for _ in range(min(5, len(indices))):
                    score, idx = heapq.heappop(indices)
                    docid = docs[idx]["docid"]
                    reference = {"score": float(-score), "content": docs[idx]['content']}
                    topk.append(docid)
                    references.append(reference)
                
                # Update response with re-ranked results
                response = {
                    "eval_id": row.get("eval_id"),
                    "standalone_query": standalone_query,
                    "topk": topk,
                    "references": references
                }
                updated_results.append(response)
                
                

                # 3-1 LLM call for further rescoring of results
                # msg = []
                # for i in range(len(response.get("topk"))):
                #     topk_docid = response.get("topk")[i]
                #     topk_content = response.get("references")[i].get("content")

                #     msg = [{"role": "user", "content": f"{topk_content} 위 지문이 {response['standalone_query']} 위 질문에 알맞는 내용이면 True, 알맞지 않으면 False라고 답해줘."}]
                                        
                #     print(f"LLM 호출을 위해 메시지 생성: {msg}")

                #     try:
                #         qaresult = client.chat.completions.create(
                #             model=llm_model,
                #             messages=msg,
                #             temperature=0,
                #             seed=1,
                #             timeout=30
                #         )
                #         print(f"LLM 응답: {qaresult.choices[0].message.content}")
                #     except Exception as e:
                #         traceback.print_exc()
                #         continue  # 현재 문서가 실패해도 다음 문서로 계속 진행

                #     # LLM의 응답이 True인 경우 response 업데이트하고 루프 탈출
                #     if qaresult.choices[0].message.content.strip().lower() == "true":
                #         response["topk"] = [topk_docid]
                #         print(f"LLM에서 True 응답을 받았으므로 문서 ID '{topk_docid}'를 선택합니다.")
                #         break

                # updated_results.append(response)

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

rescore_existing_results('/data/ephemeral/home/data/highscore.csv', "/data/ephemeral/home/sample_submission14_snunlpKLUE_combined_synonyms_reranking_jimiprompt_documeta100_top5.csv")
# MAP Score: 0.7517676767676765 MAP Score: 0.7880880230880226
#rescore_existing_results("/data/ephemeral/home/sample_submission6_roberta_sparse_fullprompt.csv", "/data/ephemeral/home/sample_submission14_roberta_hybridmodified_synonyms_reranking.csv")
# MAP Score: 0.667929292929293 hybrid
# rescore_existing_results("/data/ephemeral/home/sample_submission5_roberta_re-search.csv", "/data/ephemeral/home/sample_submission13_roberta_hybriddocid_synonyms.csv")
# MAP Score: 0.5598484848484854 sparse should
# rescore_existing_results("/data/ephemeral/home/sample_submission5_roberta_re-search.csv", "/data/ephemeral/home/sample_submission13_roberta_hybriddocid_synonyms.csv")
# MAP Score: 0.496969696969697 hybrid
# rescore_existing_results("/data/ephemeral/home/sample_submission9_roberta_hybrid_4omini.csv", "/data/ephemeral/home/sample_submission13_roberta_hybriddocid_synonyms.csv")
# MAP Score: 0.665909090909091 hybrid
# rescore_existing_results("/data/ephemeral/home/sample_submission6_roberta_llmchoice_fullprompt.csv", "/data/ephemeral/home/sample_submission13_roberta_hybriddocid_synonyms.csv")


