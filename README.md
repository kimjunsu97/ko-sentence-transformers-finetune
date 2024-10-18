# ko-sentence-transformers-finetune
이 프로젝트는 한국어 문장 임베딩 모델을 다양한 데이터셋과 방법론으로 파인튜닝하고, 그 성능을 평가하는 것을 목표로 합니다. 여러 한국어 문장 임베딩 모델을 학습하여 비교 분석하고, 각 모델이 다양한 자연어 처리(NLU) 작업에서 얼마나 우수한 성능을 보이는지 테스트합니다. 이를 통해 최적의 문장 임베딩 모델을 도출하고, 다양한 실제 애플리케이션에 적용할 수 있는 모델을 선정하는 데 목적이 있습니다.
## env 설정

```
$git clone https://github.com/kimjunsu97/ko-sentence-transformers-finetune.git
$cd ko-sentence-transformers-finetune
$pip install -r requirements.txt
```
## 데이터셋 준비 방법

### PAWS-X 데이터셋
```
$wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz
$tar -zxvf x-final.tar.gz
```
### KorNLI, KorSTS 데이터셋
```
$git clone https://github.com/kakaobrain/kor-nlu-datasets.git
```

## 모델 local 저장 방법
```
$python save_model.py
```

## 모델 train 방법
```
$python train_model.py
```

## 모델 test 방법
```
$python test_model.py
```

