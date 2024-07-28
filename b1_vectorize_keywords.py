import json
import ndjson
from transformers import BertTokenizer, BertModel
import torch
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import numpy as np  # Added import for numpy

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 디렉토리 설정
OUTPUT_DIR = 'output'
VECTORIZATION_DIR = 'b1_Vectorized'
os.makedirs(VECTORIZATION_DIR, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATABASE_FILE = os.path.join(VECTORIZATION_DIR, f'vectorized_{current_time}.ndjson')

# BERT를 사용하여 키워드 벡터화
def get_keyword_vector(keyword):
    inputs = tokenizer(keyword, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# 날짜 벡터화 (연도, 월, 일 분리)
def date_to_vector(date_str):
    year, month, day = map(int, date_str.split('-'))
    return [year, month, day]

# 순위 정규화
def normalize_rank(ranks):
    scaler = MinMaxScaler()
    return scaler.fit_transform(np.array(ranks).reshape(-1, 1))

# 고유 ID 생성 (일련번호 적용)
def generate_unique_ids(data):
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_index_map = defaultdict(int)
    
    for item in data:
        date_str = item['date']
        date_index_map[date_str] += 1
        index = date_index_map[date_str]
        item['id'] = f"{date_str}-{today}-{str(index).zfill(5)}"
    
    return data

# 벡터화 작업 함수
def process_entry(entry):
    keyword_vector = get_keyword_vector(entry["keyword"]).squeeze()
    return {"vector": keyword_vector, "date": entry["date"], "rank": entry["rank"], "keyword": entry["keyword"]}

if __name__ == '__main__':
    # 입력 폴더의 모든 파일 처리
    all_data = []
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.ndjson'):
            output_file = os.path.join(OUTPUT_DIR, filename)
            
            # 파일 읽기
            with open(output_file, 'r', encoding='utf-8') as f:
                data = ndjson.load(f)
                all_data.extend(data)

    # 멀티프로세싱 설정
    num_processes = max(cpu_count() - 1, 1)  # 최소 1개의 프로세스는 사용하도록 설정
    with Pool(num_processes) as pool:
        results = pool.map(process_entry, all_data)

    # 고유 ID 생성
    all_data = generate_unique_ids(all_data)

    # 결과 저장을 위한 데이터 생성
    database_entries = []
    for index, (result, item) in enumerate(zip(results, all_data)):
        entry = {
            "id": item['id'],
            "vector": result['vector'].tolist(),
            "date": result["date"],
            "rank": result["rank"],
            "keyword": result["keyword"]
        }
        database_entries.append(entry)

    # 데이터베이스 파일에 저장
    with open(DATABASE_FILE, 'w', encoding='utf-8') as db_file:
        ndjson.dump(database_entries, db_file)

    print(f"Vectorization completed and saved to {DATABASE_FILE}")
