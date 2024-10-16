#!/bin/bash

# Elasticsearch 8.15.2 다운로드 및 압축 해제
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.2-linux-x86_64.tar.gz
tar -xzvf elasticsearch-8.15.2-linux-x86_64.tar.gz


# daemon user로 구동을 하기 위해 소유자 변경
chown -R daemon:daemon elasticsearch-8.15.2/

# Nori 형태소 분석기 설치
./elasticsearch-8.15.2/bin/elasticsearch-plugin install analysis-nori

# Elasticsearch 데몬으로 구동 및 60초 대기
sudo -u daemon elasticsearch-8.15.2/bin/elasticsearch -d
echo "Waiting 60 seconds..."
sleep 60   

# Elasticsearch 비밀번호 변경(앞으로 사용할 비밀번호 정보를 얻기 위해 실행)
sudo -u daemon elasticsearch-8.15.2/bin/elasticsearch-reset-password -u elastic
