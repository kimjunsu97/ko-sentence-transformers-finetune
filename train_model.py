from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, models, losses
from dataset import load_kor_paws_data, load_kor_sts_data, load_kor_paws_dataset, load_kor_sts_dataset
from torch.utils.data import DataLoader
from evaluator import CustomEmbeddingSimilarityEvaluator
#from accelerate import Accelerator
#from torch.optim import AdamW

import os
import torch
import gc


import yaml
from box import Box

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    #accelerator = Accelerator()
    #device = accelerator.device

    conf_url = 'config.yaml'
    with open(conf_url, 'r') as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)



    for dataset in config.dataset_list:
        if dataset == "sts":
            train_dataset = load_kor_sts_data(config.dataset_list.sts.train)
            val_dataset = load_kor_sts_data(config.dataset_list.sts.val)
            #rain_dataset = load_kor_sts_data(config.dataset_list.sts.train)
            #val_dataset = load_kor_sts_data(config.dataset_list.sts.val)
            
        if dataset == "paws":
            train_dataset = load_kor_paws_data(config.dataset_list.paws.train)
            val_dataset = load_kor_paws_data(config.dataset_list.paws.val)
            #train_dataset = load_kor_paws_data(config.dataset_list.paws.train)
            #val_dataset = load_kor_paws_data(config.dataset_list.paws.val)

        train_dataloader= DataLoader(train_dataset,shuffle=True, batch_size=config.train_args.batch_size)
        evaluator = CustomEmbeddingSimilarityEvaluator.from_input_examples(val_dataset, batch_size = config.train_args.batch_size)

        for model_name in config.model.train_model_list:
            model = SentenceTransformer(f"./model/{model_name}",trust_remote_code=True)
            train_loss = losses.CosineSimilarityLoss(model) 
            #optimizer = AdamW(model.parameters(), lr=2e-5)
 
            #model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

            

#             args = SentenceTransformerTrainingArguments(
#                 output_dir=f"./finetune_{dataset}/{model_name}",
#                 num_train_epochs = config.train_args.epoch,
#                 per_device_train_batch_size = config.train_args.batch_size,
#                 per_device_eval_batch_size = config.train_args.batch_size,
#                 learning_rate= float(config.train_args.learning_rate),
#                 warmup_ratio=config.train_args.warmup_ratio,
#                 fp16 = config.train_args.fp16,
#                 bf16 = config.train_args.bf16,
#                 eval_strategy=config.train_args.eval_strategy,
#                 eval_steps=config.train_args.eval_steps,
#                 save_strategy=config.train_args.save_strategy,
#                 save_steps=config.train_args.save_steps
#             )

#             trainer = SentenceTransformerTrainer(
#                 model=model,
#                 args=args,
#                 train_dataset=train_dataset,
#                 #eval_dataset=val_dataloader,
#                 #loss=train_loss,
#                 #evaluator=evaluator,
# )           
#             trainer.train()

            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=config.train_args.epoch,
                #optimizer=optimizer,
                output_path=f"./finetune_{dataset}/{model_name}"
            )

if __name__ == "__main__":
    main()
    

