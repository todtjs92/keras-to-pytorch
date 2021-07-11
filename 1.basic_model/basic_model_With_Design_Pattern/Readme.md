# 파일 설명 

### mnist 데이터 이용한 Simple Dnn 모델 

mode.pth -> 저장된 모델
model.py -> 모델 정의 하는 파이썬 파일
trainier.py -> 모델 실행 및 , train ,validate 하는 코드 들어있음.
train.py -> 모델 실행 entry point 
util.py -> 데이터셋 로드 

## 파일 실행법

python3 train.py --model_fn  mode.pth 
나머지 args 는 --help 참조
 
