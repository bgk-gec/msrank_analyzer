import os
import ndjson
import numpy as np
from sklearn.cluster import KMeans
import json
from datetime import datetime
import plotly.graph_objects as go

# 디렉토리 설정
VECTOR_FILE_DIR = './c1_vectorized_cluster'
DATABASE_FILE_DIR = './b1_Vectorized'
DATABASE_FILE_PATTERN = 'vectorized_'

# 가장 최근의 파일 찾기
def get_latest_file(directory, pattern):
    files = [f for f in os.listdir(directory) if f.startswith(pattern)]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0]) if files else None

DATABASE_FILE = get_latest_file(DATABASE_FILE_DIR, DATABASE_FILE_PATTERN)

if DATABASE_FILE is None:
    raise FileNotFoundError("Could not find the required vectorized files.")

# 데이터 로드
with open(DATABASE_FILE, 'r', encoding='utf-8') as db_file:
    data = ndjson.load(db_file)

# 벡터 추출
vectors = np.array([entry['vector'] for entry in data])

# 결과 파일 경로 설정
RESULT_FILE_DIR = './c2a_chgs_by_date'
os.makedirs(RESULT_FILE_DIR, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d")

# c1_vectorized_cluster 디렉토리에서 카테고리 파일명 추출
category_files = [f for f in os.listdir(VECTOR_FILE_DIR) if f.startswith('c1_') and f.endswith('.ndjson')]
categories = [f[3:-7] for f in category_files]  # 'c1_'와 '.ndjson'을 제거하여 카테고리 이름 추출

for category, file_name in zip(categories, category_files):
    vector_file = os.path.join(VECTOR_FILE_DIR, file_name)
    with open(vector_file, 'r', encoding='utf-8') as vf:
        cluster_vectors = json.load(vf)
    
    # 클러스터 이름과 벡터 분리
    cluster_names = list(cluster_vectors.keys())
    cluster_centers = np.array([np.array(vec) for vec in cluster_vectors.values()])
    cluster_centers = np.squeeze(cluster_centers)  # 3차원 배열을 2차원으로 변환
    
    # K-means 클러스터링 수행 (사용자 지정 클러스터 중심점)
    kmeans = KMeans(n_clusters=len(cluster_names), init=cluster_centers, n_init=1)
    kmeans.fit(vectors)
    assigned_clusters = kmeans.labels_
    
    # 날짜별 클러스터 변화 추적
    date_cluster_counts = {}
    for entry, label in zip(data, assigned_clusters):
        date = entry['date']
        cluster_name = cluster_names[label]
        if date not in date_cluster_counts:
            date_cluster_counts[date] = {name: 0 for name in cluster_names}
        date_cluster_counts[date][cluster_name] += 1
    
    # 결과 파일 저장 경로 설정
    RESULT_FILE = os.path.join(RESULT_FILE_DIR, f'c2a_{category}_{current_time}.ndjson')
    
    clustered_data = [{"date": date, "clusters": clusters} for date, clusters in date_cluster_counts.items()]
    
    # 결과 저장
    with open(RESULT_FILE, 'w', encoding='utf-8') as result_file:
        ndjson.dump(clustered_data, result_file)
    
    print(f"Clustered data by date summary for '{category}' saved to {RESULT_FILE}")

    # 데이터 프레임 생성
    import pandas as pd
    df = pd.DataFrame.from_dict(date_cluster_counts, orient='index')
    df = df.sort_index()
    
    # 라인 차트 생성
    fig = go.Figure()
    
    for column in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
    
    # 라인 끝에 키워드 추가 및 겹침 방지
    annotations = []
    last_values = df.iloc[-1]
    y_positions = []
    y_shift = 1.5  # 기본 이동 거리

    for i, column in enumerate(df.columns):
        y = last_values[column]
        y_pos = y
        # 겹치는지 확인하고 이동
        while y_pos in y_positions:
            y_pos += y_shift  # 이동 거리 증가
            y_shift += 0.5  # 이동 거리 더 증가
        y_positions.append(y_pos)
        annotations.append(dict(x=df.index[-1], y=y_pos, text=column, showarrow=False))
        print(f"Adjusted position for '{column}' to {y_pos} due to overlap")
    
    fig.update_layout(
        annotations=annotations,
        title=f"Cluster Changes Over Time for {category}",
        xaxis_title="Date",
        yaxis_title="Count",
        width=1920,
        height=1080
    )
    
    # 차트 저장
    CHART_FILE = os.path.join(RESULT_FILE_DIR, f'c2a_{category}_chart_{current_time}.png')
    fig.write_image(CHART_FILE)
    
    print(f"Line chart for '{category}' saved to {CHART_FILE}")
