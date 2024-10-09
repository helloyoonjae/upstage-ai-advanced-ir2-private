import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Sentence Transformer 모델 초기화 (한국어 임베딩 생성 가능한 어떤 모델도 가능)

#model = SentenceTransformer("jhgan/ko-sroberta-multitask")
model = SentenceTransformer("/data/ephemeral/home/code/output/training_jhgan-ko-sroberta-multitask_binary_2024-10-09_15-36-16-final")
#model = SentenceTransformer("/data/ephemeral/home/code/sroberta_finetuned_epoch_1")
#model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# {"eval_id": 32, "msg": [{"role": "user", "content": "오늘 너무 즐거웠다!"}]} --> 과학상식으로 standalone_query를 뽑음
# --> 이 문제는 embedding문제가 아니라 standalone_query를 이상하게 뽑은 llm문제



# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
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

    # # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    # knn_query = {
    #     "field": "embeddings",
    #     "query_vector": query_embedding.tolist(),
    #     "k": size,
    #     "num_candidates": 200
    # }


    # sparsedocids가 있을 경우 해당 문서들로 제한하여 검색
    print("sparsedocids: ", sparsedocids)
    if sparsedocids:
        knn_query = {
            "script_score": { # 각 문서의 유사도를 점수로 매깁니다
                "query": {
                    "ids": {  # 'ids' 쿼리를 사용하여 특정 문서 ID 집합을 대상으로 설정
                        "values": sparsedocids
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                    "params": {
                        "query_vector": query_embedding.tolist()
                    }
                }
            }
        }
    else:
        # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
        knn_query = {
            "script_score": { # 각 문서의 유사도를 점수로 매깁니다
                "query": {
                    "match_all": {} # 모든 문서 검색
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
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

    # body = {
    #     "query": {
    #         "match": {
    #             "content": {
    #                 "query": query_str,
    #                 # "boost": 0.0005
    #                 "boost": 0.0025
    #             }
    #         }
    #     },
    #     "knn": {
    #         "field": "embeddings",
    #         "query_vector": query_embedding.tolist(),
    #         "k": 5,
    #         "num_candidates": 50,
    #         "boost": 1
    #     },
    #     "size": size
    # }
    
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



# 역색인 + Vector 유사도 혼합 (제출은 안했지만 안좋아 보여 폐기)
def hybrid2_retrieve(query_str, size):
    query = {
        "bool": {
            "must": [
                {"match": {"content": query_str}}
            ]
        }
    }
    must_result = es.search(index="test", 
                     query=query, 
                     size=5, 
                     sort="_score:desc",  # 점수에 따라 내림차순 정렬
                     track_total_hits=True,  # 정확한 검색 결과 총 개수 추적
                     explain=True,
                     request_cache=False                 
    )
    response = {"docids": []}
    for rst in must_result['hits']['hits']:
        response["docids"].append(rst["_source"]["docid"])
    
    query_embedding = get_embedding([query_str])[0]
    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn_query = {
        "script_score": { # 각 문서의 유사도를 점수로 매깁니다
            "query": {
                "ids": {  # 'ids' 쿼리를 사용하여 특정 문서 ID 집합을 대상으로 설정
                    "values": response["docids"]
                }
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                "params": {
                    "query_vector": query_embedding.tolist()
                }
            }
        }
    }

    
    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", query=knn_query, size=size)

from elasticsearch import Elasticsearch

es_username = "elastic"
es_password = "1oOsQPhX3G91Ogky-wul"

# Elasticsearch client 생성
es = Elasticsearch(['https://localhost:9200'], 
                   basic_auth=(es_username, es_password), 
                   ca_certs='/data/ephemeral/home/elasticsearch-8.8.0/config/certs/http_ca.crt')
                   #ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt" 상대경로 에러남)

# Elasticsearch client 정보 확인
print(es.info())

# # 노드 정보 조회
# nodes_info = es.cat.nodes(format="json")

# # 노드 정보 출력
# print(nodes_info)

#{'name': 'instance-11780', 'cluster_name': 'elasticsearch', 'cluster_uuid': 'DZFW7vCKTuak0W6cYoF-GQ', 'version': {'number': '8.8.0', 'build_flavor': 'default', 'build_type': 'tar', 'build_hash': 'c01029875a091076ed42cdb3a41c10b1a9a5a20f', 'build_date': '2023-05-23T17:16:07.179039820Z', 'build_snapshot': False, 'lucene_version': '9.6.0', 'minimum_wire_compatibility_version': '7.17.0', 'minimum_index_compatibility_version': '7.0.0'}, 'tagline': 'You Know, for Search'}
#[{'ip': '127.0.0.1', 'heap.percent': '29', 'ram.percent': '58', 'cpu': '1', 'load_1m': '2.47', 'load_5m': '1.45', 'load_15m': '1.23', 'node.role': 'cdfhilmrstw', 'master': '*', 'name': 'instance-11780'}]
    
#{'name': 'instance-11780', 'cluster_name': 'elasticsearch', 'cluster_uuid': 'DZFW7vCKTuak0W6cYoF-GQ', 'version': {'number': '8.8.0', 'build_flavor': 'default', 'build_type': 'tar', 'build_hash': 'c01029875a091076ed42cdb3a41c10b1a9a5a20f', 'build_date': '2023-05-23T17:16:07.179039820Z', 'build_snapshot': False, 'lucene_version': '9.6.0', 'minimum_wire_compatibility_version': '7.17.0', 'minimum_index_compatibility_version': '7.0.0'}, 'tagline': 'You Know, for Search'}
#[{'ip': '127.0.0.1', 'heap.percent': '31', 'ram.percent': '58', 'cpu': '4', 'load_1m': '1.52', 'load_5m': '1.36', 'load_15m': '1.21', 'node.role': 'cdfhilmrstw', 'master': '*', 'name': 'instance-11780'}]    
    
# # 샤드 정보 가져오기
# shards_info = es.cat.shards(format="json")

# # 샤드 정보 출력
# for shard in shards_info:
#     print(f"인덱스: {shard['index']}, 샤드 번호: {shard['shard']}, 상태: {shard['state']}, 노드: {shard['node']}")    
    
# 인덱스: test, 샤드 번호: 0, 상태: STARTED, 노드: instance-11780
# 인덱스: test, 샤드 번호: 0, 상태: UNASSIGNED, 노드: None
# 인덱스: your_index, 샤드 번호: 0, 상태: STARTED, 노드: instance-11780
# 인덱스: your_index, 샤드 번호: 0, 상태: UNASSIGNED, 노드: None
# 인덱스: .security-7, 샤드 번호: 0, 상태: STARTED, 노드: instance-11780    
    
    
# 인덱스: .security-7, 샤드 번호: 0, 상태: STARTED, 노드: instance-11780
# 인덱스: test, 샤드 번호: 0, 상태: STARTED, 노드: instance-11780
# 인덱스: test, 샤드 번호: 0, 상태: UNASSIGNED, 노드: None
# 인덱스: your_index, 샤드 번호: 0, 상태: STARTED, 노드: instance-11780
# 인덱스: your_index, 샤드 번호: 0, 상태: UNASSIGNED, 노드: None




                #"updateable": True
                #"lenient": True, # 오류허용
                # "synonyms": [
                #     "예외 처리, 예외처리",
                #     "Dmitri Ivanovsky, 드미트리 이바노프스키",
                #     "traffic, 트래픽",
                #     "sniffing, 스니핑",
                #     "python, 파이썬",
                #     "list, 리스트",
                #     "bridge inverter, 브리지 인버터",
                #     "inverter, 인버터",
                #     "bridge, 브리지",
                #     "lambda, 람다",
                #     "interferon, 인터페론",
                #     "operator, 연산자",
                #     "trojan, 트로이군",
                #     "sample, 샘플",
                #     "리보오솜, 리보솜, 리보좀",
                #     "디엔에이, DNA",
                #     "RNA, 리보핵산, 알엔에이, 알앤에이",
                #     "mRNA, 메시지 RNA, 메신저 RNA",
                #     "tRNA, 전달 RNA",
                #     "네트웤,네트워크,네트웍",
                #     "평형 분극 비율, 평형분극비율",
                #     "래브라도 리트리버, 리트리버",
                #     "산소, O2, O_{2}",
                #     "이산화탄소, CO2",
                #     "심층 염기서열 분석, NGS, 염기서열 분석, 염기서열분석",
                #     "frog, 개구리",
                #     "DNMT1, DNA 메틸트랜스퍼라제, 메틸트랜스퍼라제",
                #     "세계보건기구,WHO",
                #     "유순성 가설,docility hypothesis",
                #     "Basal Metabolic Rate, BMR",
                #     "메시지 인증 코드, MAC",
                #     "염소기체, 염소 기체, Cl2",
                #     "염소, CL, Cl, cl",
                #     "PAL,Programmable Array Logic",
                #     "PLD,Programmable Logic Device",
                #     "렙틴, 랩틴, Leptin",
                #     "아이오딘화납,PbI2",
                #     "FM, frequency modulation, 주파수 변조, 주파수변조",
                #     "디메틸니트록시드, (CH3)2NO",
                #     "GAL,Generic Array Logic",
                #     "MERS, 중동호흡기증후군, Middle East Respiratory Syndrome",
                #     "The Atherogenic Lipoprotein Phenotype, ALP",
                #     "The respiration quotient, RQ",
                #     "스칸듐 이온, Sc3+",
                #     "적외선, IR",
                #     "오르니틴 트랜스카비미라제,OTC",
                #     "시클로부틸 라디칼, C4H7",
                #     "ADL, 일상 생활 활동",
                #     "갭 결합, 갭결합",
                #     "잃음, 상실, 손실, 분실, 로스",
                #     "램프, 전구, 등, 광원, 전등",
                #     "전선, 와이어, 전기선, 케이블, 도선",
                #     "친환경 에너지, 햇빛 에너지, 태양 에너지, 풍력 에너지, 수력 에너지, 지열 에너지, 바이오매스 에너지, 해양 에너지, 수소 에너지",
                #     "해안가, 바다, 해변, 바닷가, 연안, 해안, 해변가",
                #     "나쁜 영향, 오염",
                #     "농장, 목장, 농원, 경작지, 농토",
                #     "팁, 방법, 노하우, 전략, 방식, 기법, 테크닉, 수단, 절차",
                #     "재배, 키우는, 키우다, 키운다, 생식, 생산, 번식, 만드는, 생성, 산출",
                #     "메탄,CH4, CH_{4}",
                #     "경향, 추세, 성향, 흐름, 방향",
                #     "이유, 원인, 근거, 동기, 사유, 계기, 때문",
                #     "확률, 경향, 가능성, 기회, 비율, 유력",
                #     "무엇, 정의, 의미, 해석, 설명, 뜻합니다",
                #     "짚신벌레, 짚신 벌레",
                #     "온난전선, 온난 전선",
                #     "날씨, 기후",
                #     "근친상간, 근친간의 성행위, 근친간 성행위, 근친 성행위, 근친교배, 인세스트",
                #     "측정, 분석, 계측, 평가",
                #     "중등학교, 중고등학교",
                #     "남미, 콜롬비아, 페루, 우루과이, 코스타리카",
                #     "서식지, 거주지, 번식지, 생태계, 영역",
                #     "이주, 이동",
                #     "낭비, 고갈, 소모, 소진, 말라버림, 동나감",
                #     "색깔, 색상, 컬러",
                #     "다음세대, 자손, 후손, 2세",
                #     "이슈, 문제",
                #     "음파, 소리, 사운드, 음향, 진동",
                #     "이로운, 장점, 좋은점, 이점, 좋은 방법, 선기능, 순기능,",
                #     "약,약물",
                #     "문제, 부작용",
                #     "세기, 강도",
                #     "자석, 자성체",
                #     "가루, 분말",
                #     "파악이 안된, 미지, 모르는",
                #     "옮기는, 운반",
                #     "빛, 광자",
                #     "원핵세포, 원핵 세포",   
                #     "소립자, 기본 입자, 미립자, 기본입자",
                #     "생기는, 형성, 발생, 현상",
                #     "가장 작은 값, 최소값",
                #     "침샘, 침",
                #     "과일 샐러드, 과일샐러드",
                #     "네트워크 통신 주소, ip, IP",
                #     "이탈리아, 이태리",
                #     "미디어, 언론",
                #     "칼리시바이러스, 칼리시 바이러스",
                #     "B형 간염, B-형 간염",
                #     "연령, 나이, 연세",
                #     "결혼 전, 혼전",
                #     "조심, 주의",
                #     "완구, 장난감",
                #     "움직이다, 작동, 동작, 움직이게, 움직이는",
                #     "초콜렛, 초코렛, 초콜릿",
                #     "에너지원, 에너지 소스",
                #     "감정적인 지원, 지지, 응원, 위로",
                #     "아세틸콜린, 아세틸 콜린",
                #     "남녀관계, 연애관계",
                #     "매연, 대기오염, 배기가스, 공해, 유해가스",
                #     "카탈로그, catalog",
                #     "class, 클래스"                 
   
                # ],
# "user_dictionary": ["오토마톤","NMR","Elodea","AIDS","마이애미파랑나비","헤르페스",
                #                     "광자","아메리카 알리게이터","Nmap","웹 프록시","알파 붕괴",
                #                     "찰스 다윈","니크롬선","후마네 비떼","MSM","트레드밀","ALU","CPU",
                #                     "어큐뮬레이터","SCR","인덕터","크레아틴","캐비티","ADP","Ca","사스",
                #                     "알짜이온","C6H14","OpenSSL","mountain chorus frog","베이츠 의태",
                #                     "AM","FM","유미지립","HDL","ATP","Elaphe obsoleta","쥐 뱀","GDP","DNMT1",
                #                     "튀르키 예인","NonNicks","SCR","알캅톤뇨증","백색증","후를러 증후군","관상 봉합",
                #                     "한수교","멤브레인","핵종","파울리의 배타 원리","인스턴트 메시징 애플리케이션","호이겐스의 원리",
                #                     "윈 브리지 발진기","키메라증","역 U 가설","PRP","연기 열수공","머클-담고르 해시 함수",
                #                     ],
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
            "dims": 768,
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
with open("/data/ephemeral/home/data/documents.jsonl") as f:
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


# # 기존 CSV 파일을 사용하여 재검색 수행
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
                # sparse_search_result = sparse_retrieve(standalone_query, 20)
                # sparsedocids = [doc['_id'] for doc in sparse_search_result['hits']['hits']]
                # search_result = dense_retrieve(standalone_query, 3, sparsedocids)
                
                # sparse_search_result = sparse_retrieve(standalone_query, 3)
                # sparsedocids = [doc['_id'] for doc in sparse_search_result['hits']['hits']]
                # dense_search_result = dense_retrieve(standalone_query, 3)
                # denseedocids = [doc['_id'] for doc in dense_search_result['hits']['hits']]
                # overlapping_docids = set(sparsedocids).intersection(set(denseedocids))
                # if len(overlapping_docids) >= 1:
                #     search_result = sparse_search_result
                # else:
                #     search_result = dense_search_result

                #search_result = sparse_retrieve(standalone_query, 3) # MUST # match_phrase
               
                #search_result = dense_retrieve(standalone_query, 3)
                
                search_result = hybrid_retrieve(standalone_query, 10)
                
                response = {"eval_id": row.get("eval_id"), "standalone_query": standalone_query, "topk": [], "references": []}
                for rst in search_result['hits']['hits']:
                    response["topk"].append(rst.get("_source").get("docid"))
                    response["references"].append({"score": rst.get("_score"), "content": rst.get("_source").get("content")})
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
rescore_existing_results("/data/ephemeral/home/sample_submission6_roberta_sparse_fullprompt.csv", "/data/ephemeral/home/sample_submission10_robertatrained_hybrid.csv")
#rescore_existing_results("/data/ephemeral/home/sample_submission9_roberta_hybrid_4omini.csv", "/data/ephemeral/home/sample_submission9_roberta_dense_4omini.csv")


