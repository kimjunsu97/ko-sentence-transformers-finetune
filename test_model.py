from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from evaluator import CustomEmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer
from dataset import load_kor_sts_data, load_kor_paws_data
import pandas as pd
from tqdm import tqdm

import yaml
from box import Box

conf_url = 'config.yaml'
with open(conf_url, 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

model_list = config.model.test_model_list


df = pd.DataFrame(columns=['model','finetune_epoch','dataset','pearson_cosine', 'spearman_cosine', "auc_roc", "accuracy", "f1", "precision", "recall"])
for dataset in config.dataset_list: 
    if dataset == "sts":
        test_dataset = load_kor_sts_data(config.dataset_list.sts.test)
        
    if dataset == "paws":
        test_dataset = load_kor_paws_data(config.dataset_list.paws.test)

    test_evaluator = CustomEmbeddingSimilarityEvaluator.from_input_examples(test_dataset, batch_size=config.test_args.batch_size)

    for model_name in tqdm(model_list):
        model = SentenceTransformer(f"./model/{model_name}",trust_remote_code=True)
        row = pd.DataFrame([test_evaluator(model)])
        row['model'] = model_name
        row['finetune_epoch'] = 0
        row['dataset'] = dataset
        df = pd.concat([df,row],ignore_index=False)
        df = df.sort_values(['dataset','pearson_cosine','spearman_cosine'],ascending=False)
df.to_csv(f"./results/{config.test_args.output_file}",index=False)