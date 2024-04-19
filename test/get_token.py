import torch
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
model_id = 'CompVis/stable-diffusion-v1-4'
name =''
for x in  model_id.split('/'):
    name += x+'_'
pipeline = StableDiffusionPipeline.from_pretrained(model_id)
token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
token_embeds = token_embeds.numpy()

mean_token = np.mean(token_embeds,axis=0)
std_token = np.std(token_embeds,axis=0)
standard_token = (token_embeds-mean_token)/std_token
b = np.linalg.norm(standard_token,axis=1)
# nums = 0
# for i,x in enumerate(b):
#     if x<1100:
#         nums+=1
# print(nums)
# X_scaler = StandardScaler()
# standard_token = X_scaler.fit_transform(token_embeds)
# mean = X_scaler.mean_
# var = X_scaler.var_
n = 768
pca = PCA(n_components=n)
pca.fit(standard_token)
X_token = pca.transform(standard_token)
features = pca.components_

mean_token = torch.from_numpy(mean_token)
std_token = torch.from_numpy(std_token)
features = torch.from_numpy(features)
token_dict ={
    'mean':mean_token,
    'std':std_token,
    'features':features
}
save_path = os.path.join('token_dict', name+f"pca_{n}.pt")
torch.save(token_dict, save_path)