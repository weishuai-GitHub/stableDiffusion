export DATA_DIR="datasets/Mixed_dataset"
ROOT="/opt/data/private/stable_diffusion_model"
DIR="textual_inversion_find_xl_mixed_768_1"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
accelerate launch sdxl_main.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --save_steps=2000 \
  --checkpointing_steps=2500 \
  --placeholder_token="#" --initializer_token="object" \
  --resolution=768 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5500 \
  --learning_rate=1.0e-04 --scale_lr \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" \
  --dataloader_num_workers=0 \
  --lr_warmup_steps=0 \
  --out_dim=768 \
  --cls_dim=10 \
  --repeats 100 \
  --unlabelled 11 \
  --style_name "Ande Cubism Cute Fauvism Landscape_painting" \
  --output_dir=${ROOT}/${DIR}
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export DATA_DIR="./bird"

# accelerate launch textual_inversion_sdxl.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token="<cat-toy>" \
#   --initializer_token="toy" \
#   --mixed_precision="bf16" \
#   --resolution=768 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=500 \
#   --learning_rate=5.0e-04 \
#   --scale_lr \
#   --validation_prompt="a <cat-toy>" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --resume_from_checkpoint='latest' \
#   --output_dir="./textual_inversion_bird_sdxl"
