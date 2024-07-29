import json
from transformers import BertTokenizer, BertModel
import torch
import os
print(torch.__path__)

# 클러스터 기준 설정 (영어로 변환된 키워드들)
clusters = {
    "Style": ["Americana Casual", "Athleisure", "Business", "Casual", "Chic", "Classic", "Contemporary", "Country", "Formal", "Gorpcore", "Grandma Chic", "Loose", "Minimalism", "Military",  "Preppy", "Retro",  "Street", "Tomboy", "Traditional", "Urban", "Vintage"],
    "Type": ["t-shirt", "dress", "jeans", "coat", "jacket", "skirt", "blouse", "cardigan", "sweater", "shorts", "pants", "leggings", "suit", "blazer", "camisole", "tunic", "parka", "vest", "anorak"],
    "Season": ["summer", "winter", "spring", "autumn", "rainy", "hot", "cold", "windy", "humid", "snowy", "stormy", "fog", "mild weather", "transitional season", "hiking"],
    "Items": ["bag", "shoes", "hat", "accessory", "watch", "jewelry", "belt", "scarf", "gloves", "sunglasses", "wallet", "brooch", "hair accessory", "tie", "cufflinks"],
    "Color": ["black", "white", "red", "blue", "green", "pink", "yellow", "purple", "orange", "brown", "gray", "beige", "maroon", "navy", "teal", "Monochrome", "turquoise", "coral", "olive", "peach", "lavender"],
    "Material": ["cotton", "linen", "denim", "leather", "wool", "silk", "polyester", "rayon", "nylon", "velvet", "satin", "chiffon", "lace", "cashmere", "spandex", "tweed", "hemp", "modal", "suede", "tulle"],
    "Trends": ["trend", "Prints", "SNS", "YouTube", "celebrity", "viral", "runway", "fashion week", "eco-friendly", "sustainable", "upcycling", "gender-neutral", "athleisure", "minimalism", "maximalism", "micro", "airport"],
    "Issues": ["environmental", "protection", "sustainability", "ethical", "social responsibility", "diversity", "fair", "transparency", "animal welfare", "body", "positivity"],
    "Events": ["wedding", "party", "travel", "vacation", "Christmas", "Halloween", "New Year", "Valentine's Day", "birthday", "graduation", "anniversary", "holiday", "Buddha", ]
}

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().tolist()

# 벡터화된 클러스터 기준 저장 경로 설정
VECTOR_FILE_DIR = './c1_vectorized_cluster'
os.makedirs(VECTOR_FILE_DIR, exist_ok=True)

# 각 대분류에 대해 벡터화된 클러스터 기준 저장
for category, keywords in clusters.items():
    cluster_vectors = {}
    for keyword in keywords:
        cluster_vectors[keyword] = embed_text(keyword)
    
    VECTOR_FILE = os.path.join(VECTOR_FILE_DIR, f'c1_{category.lower()}.ndjson')
    
    with open(VECTOR_FILE, 'w', encoding='utf-8') as vector_file:
        json.dump(cluster_vectors, vector_file)
    
    print(f"Cluster vectors for '{category}' saved to {VECTOR_FILE}")
