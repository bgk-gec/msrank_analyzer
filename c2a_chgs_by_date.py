import os
import ndjson
import numpy as np
from sklearn.cluster import KMeans
import json
from datetime import datetime
from scipy.spatial.distance import cosine  # 코사인 유사도 계산을 위해 추가
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

# 유사성 임계값 설정
SIMILARITY_THRESHOLD = 0.75

# 폰트 크기 설정
font_size = 14

# 모든 카테고리의 데이터를 저장할 리스트
all_category_top5_data = []

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
        similarity = 1 - cosine(entry['vector'], cluster_centers[label])
        if similarity >= SIMILARITY_THRESHOLD:
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
    y_positions = set()
    x_positions = []

    x_values = list(df.index)
    x_adjustment = int(len(x_values) * 0.05)  # x 좌표 조정 비율 계산

    # 1차: y=0이 아닌 경우 y 좌표 조정
    for i, column in enumerate(df.columns):
        y = last_values[column]
        x = df.index[-1]

        if y != 0:
            while (x, y) in y_positions:
                y -= 1

        y_positions.add((x, y))
        if y == 0:
            x_positions.append((x, y, column))  # y=0인 경우 x 좌표 조정을 위해 따로 저장
        else:
            annotations.append(dict(x=x, y=y, text=column, showarrow=False, font=dict(size=font_size)))
            print(f"Adjusted position for '{column}' to (x: {x}, y: {y})")

    # 2차: y=0인 경우 x 좌표 조정
    for (x, y, column) in x_positions:
        while (x, y) in y_positions:
            x_index = x_values.index(x)
            adjusted_x_index = x_index - x_adjustment
            if adjusted_x_index >= 0:
                x = x_values[adjusted_x_index]
            else:
                break  # x 좌표 조정이 더 이상 불가능한 경우

        y_positions.add((x, y))
        annotations.append(dict(x=x, y=y, text=column, showarrow=False, font=dict(size=font_size)))
        print(f"Adjusted x position for '{column}' to (x: {x}, y: {y})")
    
    fig.update_layout(
        annotations=annotations,
        title=f"Cluster Changes Over Time for {category}",
        xaxis_title="Date",
        yaxis_title="Count",
        width=1920,
        height=1080,
        font=dict(size=font_size)
    )
    
    # 차트 저장
    CHART_FILE = os.path.join(RESULT_FILE_DIR, f'c2a_{category}_chart_{current_time}.png')
    fig.write_image(CHART_FILE)
    
    print(f"Line chart for '{category}' saved to {CHART_FILE}")

    # 클러스터별 상위 5개 키워드 저장
    sorted_clusters = sorted(date_cluster_counts[sorted(date_cluster_counts.keys())[-1]].items(), key=lambda item: item[1], reverse=True)
    category_top5_data = [f"## {category}"]
    for rank, (keyword, count) in enumerate(sorted_clusters[:5], start=1):
        category_top5_data.append(f"{rank}. {keyword} - {count} count")
    category_top5_data.append("")  # 한 카테고리가 끝난 후 빈 줄 추가
    
    all_category_top5_data.extend(category_top5_data)
    
# 최종적으로 모든 카테고리의 데이터를 한 번에 텍스트 파일로 저장
top5_file_path = os.path.join(RESULT_FILE_DIR, f'c2a_top5_{current_time}.txt')
with open(top5_file_path, 'w', encoding='utf-8') as top5_file:
    for line in all_category_top5_data:
        top5_file.write(line + "\n")

print(f"Top 5 data for each category saved to {top5_file_path}")
