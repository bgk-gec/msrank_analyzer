import ndjson
from collections import defaultdict

# ndjson 파일 불러오기
file_path = './c2_clustered_keywords/c2a_20240728_182356.ndjson'
with open(file_path) as f:
    data = ndjson.load(f)

# 데이터 구조 확인을 위해 일부 데이터 출력
print(data[:2])

# 클러스터 별로 키워드 추출
keyword_clusters = defaultdict(set)

for entry in data:
    clusters = entry.get('clusters', {})
    for cluster, keywords in clusters.items():
        if isinstance(keywords, dict):  # keywords가 dict 타입인지 확인
            for keyword in keywords.keys():
                keyword_clusters[keyword].add(cluster)
        else:
            print(f"Unexpected data format in cluster '{cluster}': {keywords}")

# 중복 키워드 확인
duplicate_keywords = {k: v for k, v in keyword_clusters.items() if len(v) > 1}

# 결과 출력
if duplicate_keywords:
    print("중복된 키워드와 해당 클러스터들:")
    for keyword, clusters in duplicate_keywords.items():
        print(f"키워드: {keyword}, 클러스터들: {clusters}")
else:
    print("중복된 키워드가 없습니다.")
