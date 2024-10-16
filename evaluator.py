from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

import numpy as np
from scipy.stats import pearsonr, spearmanr

class CustomEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # 한 번만 임베딩 계산
        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, convert_to_numpy=True)

        # 기존 거리 및 유사성 계산
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

        # 평가 결과 계산 (Pearson 및 Spearman)
        eval_pearson_cosine, _ = pearsonr(self.scores, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(self.scores, cosine_scores)
        
        threshold = 0.5

        # 예측값(이진 분류로 변환)
        predicted_labels = (cosine_scores >= threshold).astype(int)  # 임계값을 기준으로 이진 분류

        # AUC-ROC는 확률 점수를 사용
        eval_auc_roc = roc_auc_score(list(map(int,self.scores)), cosine_scores)
        
        # F1, Precision, Recall, Accuracy는 이진 분류 결과를 사용
        eval_accuracy = accuracy_score(list(map(int,self.scores)), predicted_labels)
        eval_f1 = f1_score(list(map(int,self.scores)), predicted_labels)
        eval_precision = precision_score(list(map(int,self.scores)), predicted_labels)
        eval_recall = recall_score(list(map(int,self.scores)), predicted_labels)

        # 결과 저장
        metrics = {
            "pearson_cosine": eval_pearson_cosine,
            "spearman_cosine": eval_spearman_cosine,
            "auc_roc": eval_auc_roc,
            "accuracy": eval_accuracy,
            "f1": eval_f1,
            "precision": eval_precision,
            "recall": eval_recall,
            
        }

        # 결과 반환
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics)
        return metrics