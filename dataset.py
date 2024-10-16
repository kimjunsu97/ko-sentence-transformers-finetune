import pandas as pd
from sentence_transformers import InputExample
from datasets import Dataset


def load_kor_sts_data(filename):
    # 파일을 pandas DataFrame으로 읽기
    df = pd.read_csv(filename, delimiter='\t', encoding='utf8', on_bad_lines='skip')
    df = df.dropna()
    data = []
    # 각 행을 반복하면서 필요한 데이터를 추출
    for _, row in df.iterrows():
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        data.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    
    return data

def load_kor_paws_data(filename):
    # 파일을 pandas DataFrame으로 읽기
    df = pd.read_csv(filename, delimiter='\t', encoding='utf8', on_bad_lines='skip')
    df = df.dropna()
    data = []
    # 각 행을 반복하면서 필요한 데이터를 추출
    for _, row in df.iterrows():
        label = row['label']
        data.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label))
    
    return data
    
def load_kor_sts_dataset(filename):
    # 파일을 pandas DataFrame으로 읽기
    df = pd.read_csv(filename, delimiter='\t', encoding='utf8', on_bad_lines='skip')
    df = df.dropna()
    
    dataset = {
        'sentence1': df['sentence1'].tolist(),
        'sentence2': df['sentence2'].tolist(),
        'label': df['score'].tolist()
    }
    return Dataset.from_dict(dataset)

def load_kor_paws_dataset(filename):
    # 파일을 pandas DataFrame으로 읽기
    df = pd.read_csv(filename, delimiter='\t', encoding='utf8', on_bad_lines='skip')
    df = df.dropna()

    dataset = {
        'sentence1': df['sentence1'].tolist(),
        'sentence2': df['sentence2'].tolist(),
        'label': df['label'].tolist()
    }
    return Dataset.from_dict(dataset)