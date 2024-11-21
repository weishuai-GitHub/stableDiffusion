import torch
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
name =''
for x in  model_id.split('/'):
    name += x+'_'
pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)
token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
token_embeds_2 = pipeline.text_encoder_2.get_input_embeddings().weight.data
n = 768

def get_token_embeds(token_embeds,n = 768):
    token_embeds = token_embeds.numpy()

    norm_embeds = np.linalg.norm(token_embeds,axis=1)
    index = norm_embeds > 0
    token_embeds = token_embeds[index]
    print(f"token_embeds:{token_embeds.shape}")
    mean_token = np.mean(token_embeds,axis=0)
    std_token = np.std(token_embeds,axis=0)
    standard_token = (token_embeds-mean_token)/std_token

    b = np.linalg.norm(standard_token,axis=1)
    print(f"mean:{np.mean(b)},std:{np.std(b)}")

    pca = PCA(n_components=n)
    pca.fit(standard_token)
    X_token = pca.transform(standard_token)
    b_1 = np.linalg.norm(X_token,axis=1)
    print(f"mean:{np.mean(b_1)},std:{np.std(b_1)}")
    features = pca.components_

    mean_token = torch.from_numpy(mean_token)
    std_token = torch.from_numpy(std_token)
    features = torch.from_numpy(features)
    return mean_token,std_token,features

mean_token,std_token,features = get_token_embeds(token_embeds,n = 768)
# # spcal handle

save_path = os.path.join('token_dict', name+f"pca_{n}.pt")
token_dict_1 = torch.load(save_path)
mean_token = token_dict_1['mean']
std_token = token_dict_1['std']
features = token_dict_1['features']

mean_token_2,std_token_2,features_2 = get_token_embeds(token_embeds_2,n = 960)

token_dict ={
    'mean':mean_token,
    'std':std_token,
    'features':features,
    'mean_2':mean_token_2,
    'std_2':std_token_2,
    'features_2':features_2
}
torch.save(token_dict, save_path)