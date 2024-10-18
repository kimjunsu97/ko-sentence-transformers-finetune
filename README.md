# ko-sentence-transformers-finetune
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

