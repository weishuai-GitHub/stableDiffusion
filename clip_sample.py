from transformers import CLIPModel,CLIPTokenizer
from PIL import Image
import torch
import numpy as np


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
text = ["a painting of a lion eating a burger"]
image = Image.open("A painting of a lion eating a burger.png")
image = image.resize((224, 224))
image = np.array(image)
image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
image_features = model.encoder
clip_tokrnizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_token  = clip_tokrnizer(text)
text_features = model.encode_text(text_token, convert_to_tensor=True)
logits_per_image, logits_per_text = model(image, text)
probs = logits_per_image.softmax(dim=-1).cpu().numpy()
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]