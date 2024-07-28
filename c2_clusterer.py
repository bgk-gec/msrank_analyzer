import os
import ndjson
import numpy as np
from sklearn.cluster import KMeans
import json
import plotly.graph_objects as go
from datetime import datetime

# Plotly가 설치되지 않은 경우 설치      **c1의 결과에 대한 트리맵, 모든 데이터에 대함.**
try:
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "plotly"])
    import plotly.graph_objects as go

# 클러스터 기준 벡터 파일 경로 설정
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

# 벡터화된 클러스터 기준 로드
def load_cluster_vectors(file_path):
    with open(file_path, 'r', encoding='utf-8') as vector_file:
        return json.load(vector_file)

# 데이터 로드
with open(DATABASE_FILE, 'r', encoding='utf-8') as db_file:
    data = ndjson.load(db_file)

# 벡터 추출
vectors = np.array([entry['vector'] for entry in data])

# 결과 파일 경로 설정
RESULT_FILE_DIR = './c2_clustered_keywords'
os.makedirs(RESULT_FILE_DIR, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d")

# c1_vectorized_cluster 디렉토리에서 카테고리 파일명 추출
category_files = [f for f in os.listdir(VECTOR_FILE_DIR) if f.startswith('c1_') and f.endswith('.ndjson')]
categories = [f[3:-7] for f in category_files]  # 'c1_'와 '.ndjson'을 제거하여 카테고리 이름 추출

for category, file_name in zip(categories, category_files):
    vector_file = os.path.join(VECTOR_FILE_DIR, file_name)
    cluster_vectors = load_cluster_vectors(vector_file)
    
    # 클러스터 이름과 벡터 분리
    cluster_names = list(cluster_vectors.keys())
    cluster_centers = np.array([np.array(vec) for vec in cluster_vectors.values()])
    cluster_centers = np.squeeze(cluster_centers)  # 3차원 배열을 2차원으로 변환
    
    # K-means 클러스터링 수행 (사용자 지정 클러스터 중심점)
    kmeans = KMeans(n_clusters=len(cluster_names), init=cluster_centers, n_init=1)
    kmeans.fit(vectors)
    assigned_clusters = kmeans.labels_
    
    # 클러스터별 키워드 수집
    cluster_counts = {name: 0 for name in cluster_names}
    for label in assigned_clusters:
        cluster_counts[cluster_names[label]] += 1
    
    clustered_data = [{"cluster": name, "count": count} for name, count in cluster_counts.items()]
    
    # 결과 파일 경로 설정 및 저장
    result_file_name = f'c2_{category}_{current_time}.ndjson'
    result_file_path = os.path.join(RESULT_FILE_DIR, result_file_name)
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        ndjson.dump(clustered_data, result_file)
    
    print(f"Clustered data summary for '{category}' saved to {result_file_path}")
    
    # 트리맵 데이터 생성
    sizes = list(cluster_counts.values())
    labels = [f"{name}" for name in cluster_counts.keys()]
    values = [count for count in cluster_counts.values()]
    
    # 트리맵 시각화 및 저장
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        textinfo="label+value",
        marker=dict(line=dict(width=0.5, color='black'))
    ))
    
    fig.update_layout(
        margin=dict(t=70, l=25, r=25, b=25),
        title=f'{category.capitalize()} Cluster Distribution Treemap',
        width=1920,
        height=1080,
        font=dict(size=30)  # 폰트 크기 조정
    )

    chart_file_name = f'c2_{category}_chart_{current_time}.png'
    chart_file_path = os.path.join(RESULT_FILE_DIR, chart_file_name)
    fig.write_image(chart_file_path)
    
    print(f"Treemap for '{category}' saved to {chart_file_path}")
