import os
import ndjson
import numpy as np

# 데이터 파일 경로 설정
DATABASE_FILE = './b1_Vectorization/database.ndjson'

# 데이터 로드
with open(DATABASE_FILE, 'r', encoding='utf-8') as db_file:
    data = ndjson.load(db_file)

# 벡터 추출
vectors = np.array([entry['vector'] for entry in data])

# 벡터의 차원 수 확인
vector_dimension = vectors.shape[1]

print(f"The dimension of the vectors in the database is: {vector_dimension}")
