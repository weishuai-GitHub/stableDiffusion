# stabilityai/stable-diffusion-2-1
# stabilityai/stable-diffusion-2-1-base
# stabilityai/sd-turbo
# runwayml/stable-diffusion-v1-5
# CompVis/stable-diffusion-v1-4
# stabilityai/stable-diffusion-xl-base-1.0
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="datasets/style_datasets"
ROOT="/opt/data/private/stable_diffusion_model"
DIR="textual_inversion_find_mixed_768_1_5"
CUDA_VISIBLE_DEVICES=0,1 
accelerate launch main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --save_steps=2000 \
  --checkpointing_steps=5000 \
  --placeholder_token="#" --initializer_token="Realism" \
  --resolution=512 \
  --train_batch_size=6 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=5.0e-05 --scale_lr \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" \
  --dataloader_num_workers=0 \
  --lr_warmup_steps=0 \
  --out_dim=768 \
  --cls_dim=5 \
  --repeats 100 \
  --unlabelled 11 \
  --style_name "Ande Cubism Cute Fauvism Landscape_painting" \
  --output_dir=${ROOT}/${DIR}

# export MODEL_NAME=stabilityai/stable-diffusion-2-1-base
# export DATA_DIR="datasets/Mixed_dataset"
# ROOT="/opt/data/private/stable_diffusion_model"
# DIR="textual_inversion_find_mixed_864_base_2_1"
# # CUDA_VISIBLE_DEVICES=0,1,2,3 
# accelerate launch main.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --save_steps=2000 \
#   --checkpointing_steps=5000 \
#   --placeholder_token="#" --initializer_token="object" \
#   --resolution=512 \
#   --train_batch_size=6 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=10000 \
#   --learning_rate=1.0e-05 --scale_lr \
#   --enable_xformers_memory_efficient_attention \
#   --lr_scheduler="constant" \
#   --dataloader_num_workers=0 \
#   --lr_warmup_steps=0 \
#   --out_dim=864 \
#   --cls_dim=10 \
#   --repeats 100 \
#   --unlabelled 11 \
#   --style_label 11 \
#   --output_dir=${ROOT}/${DIR}