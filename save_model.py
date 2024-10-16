import yaml
from box import Box

conf_url = 'config.yaml'
with open(conf_url, 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

from sentence_transformers import SentenceTransformer

for model_name in config.model.save_model_list:
    try:
        cache_folder=f"./model_cache/{model_name}"
        save_folder = f"./model/{model_name}"
        model = SentenceTransformer(model_name, cache_folder=cache_folder, trust_remote_code =True)
        model.save(save_folder)
    except:
        print(f"{model_name} is unavailable")