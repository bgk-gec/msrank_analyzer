import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 가장 최근의 ndjson 파일 찾기
list_of_files = glob.glob('./c2_clustered_keywords/c2a_*.ndjson')
latest_file = max(list_of_files, key=os.path.getctime)

# NDJSON 파일 불러오기
data = []

with open(latest_file, 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line))

# 데이터 프레임으로 변환
df = pd.DataFrame(data)

# clusters 열을 개별 클러스터로 확장
expanded_data = []
for _, row in df.iterrows():
    date = row['date']
    clusters = row['clusters']
    for category, count in clusters.items():
        expanded_data.append({
            'date': date,
            'category': category,
            'count': count
        })

expanded_df = pd.DataFrame(expanded_data)

# 데이터 구조 확인
print(expanded_df.head())

# 필요한 열이 있는지 확인하고 없으면 에러 메시지 출력
required_columns = ['date', 'category', 'count']
missing_columns = [col for col in required_columns if col not in expanded_df.columns]

if missing_columns:
    raise KeyError(f"Missing columns in the data: {missing_columns}")

# 클러스터 종류 확인
clusters = expanded_df['category'].unique()

# 여러 클러스터의 변화를 한 그래프에 시각화
plt.figure(figsize=(14, 8))

for cluster in clusters:
    cluster_df = expanded_df[expanded_df['category'] == cluster]
    plt.plot(cluster_df['date'], cluster_df['count'], label=cluster)

plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Cluster Changes Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
