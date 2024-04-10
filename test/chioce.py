from PIL import Image
import os
import random

name = ''
data_dir = "/opt/data/private/datasets/wikiart/"
output_dir = "/opt/data/private/code/stableDiffusion/dataset_mixed"
name_list = [name for name in os.listdir(output_dir)]

for name in name_list:
    if name in ['Minimalism','Impressionism','Romanticism','Rococo','Realism','Expressionism','Cubism','Fauvism','Baroque']: continue
    data_dir_name = os.path.join(data_dir,name)
    print(data_dir_name)
    image_files = [f for f in os.listdir(data_dir_name) if f.endswith('.jpg')]
    if len(image_files) < 100:
        selected_images = image_files
    else:selected_images = random.sample(image_files, 1000)
    for image in selected_images:
        image_path = os.path.join(data_dir_name, image)
        img = Image.open(image_path)
        img.save(os.path.join(output_dir, name, image))
