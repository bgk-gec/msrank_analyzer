import os
import torch
# torch 설치 경로 확인
torch_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
print(torch_path)

# 시스템 환경 변수에 추가
os.environ['PATH'] += os.pathsep + torch_path
