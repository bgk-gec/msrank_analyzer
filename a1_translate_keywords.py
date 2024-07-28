import os
import json
import ndjson
from google.cloud import translate_v2 as translate
from get_oauth import authenticate

# 디렉토리 설정
INPUT_DIR = 'a0_input'
OUTPUT_DIR = 'a1_translated'

# 번역 함수 정의
def translate_text(text, target='en'):
    translate_client = translate.Client(credentials=authenticate())
    result = translate_client.translate(text, target_language=target)
    return result['translatedText']

# 입력 폴더의 모든 파일 처리
for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.ndjson'):
        input_file = os.path.join(INPUT_DIR, filename)
        output_file = os.path.join(OUTPUT_DIR, filename)
        
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            data = ndjson.load(f)

        # '키워드' 필드를 번역
        for entry in data:
            if 'keyword' in entry:
                entry['keyword'] = translate_text(entry['keyword'])

        # 번역된 데이터 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            ndjson.dump(data, f)

        print(f"Translation completed and saved to {output_file}")
