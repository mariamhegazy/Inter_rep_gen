export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-T2V-A14B"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora.py \
  --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=50 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="low" \
  --train_mode="normal" \
  --low_vram 


# accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora.py \
#   --config_path="config/wan2.2/wan_civitai_5b.yaml" \
#   --pretrained_model_name_or_path="models/Wan2.2-TI2V-5B" \
#   --train_data_dir="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/clips_short/clips_short" \
#   --train_data_meta="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/training_brief.json" \
#   --image_sample_size=1024 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=2 \
#   --video_sample_n_frames=121 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=500 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir/wan2.2_5b_finetune_ultravideo" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --enable_bucket \
#   --uniform_sampling \
#   --boundary_type="low" \
#   --train_mode="normal" \
#   --tracker_project_name "wan2.2_5b_finetune" \
#   --report_to "wandb" 


  accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora_t2a.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path="models/Wan2.2-TI2V-5B" \
  --train_data_dir="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/clips_short/clips_short" \
  --train_data_meta="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/training_brief.json" \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=121 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=1 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="output_dir/debug" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="low" \
  --train_mode="normal" \
  --tracker_project_name "debug" \
  --sparse_time_mode "start" \
  --report_to "wandb" \
  --rank 128 \
  --network_alpha 64



  accelerate launch  --mixed_precision="bf16" scripts/wan2.2/train_lora_inp.py \
    --config_path="config/wan2.2/wan_civitai_t2v.yaml" \
    --pretrained_model_name_or_path="models/Wan2.2-T2V-A14B" \
    --train_data_dir="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/clips_short/clips_short" \
    --train_data_meta="/capstor/store/cscs/swissai/a144/datasets/UltraVideo/training_brief.json" \
    --image_sample_size=1024 \
    --video_sample_size=256 \
    --token_sample_size=512 \
    --video_sample_stride=2 \
    --video_sample_n_frames=81 \
    --train_batch_size=1 \
    --video_repeat=1 \
    --gradient_accumulation_steps=1 \
    --dataloader_num_workers=8 \
    --num_train_epochs=10 \
    --checkpointing_steps=500 \
    --learning_rate=1e-04 \
    --seed=42 \
    --output_dir="output_dir/debug" \
    --gradient_checkpointing \
    --mixed_precision="bf16" \
    --adam_weight_decay=3e-2 \
    --adam_epsilon=1e-10 \
    --vae_mini_batch=1 \
    --max_grad_norm=0.05 \
    --enable_bucket \
    --uniform_sampling \
    --boundary_type="low" \
    --train_mode="ti2v" \
    --tracker_project_name "wan2.2_finetune_INP" \
    --report_to "wandb" \
    --rank 256 \
    --network_alpha 128 \
    --lora_init_path "output_dir/wan2.2_14b_finetune_ultravideo_T2A_start_last_rank256_inp_high/checkpoint-5500.safetensors" \
  


# The Training Shell Code for Image to Video
# You need to use "config/wan2.2/wan_civitai_i2v.yaml"
# 
# export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-I2V-A14B"
# export DATASET_NAME="datasets/internal_datasets/"
# export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# # export NCCL_IB_DISABLE=1
# # export NCCL_P2P_DISABLE=1
# NCCL_DEBUG=INFO

# accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_lora.py \
#   --config_path="config/wan2.2/wan_civitai_i2v.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --image_sample_size=1024 \
#   --video_sample_size=256 \
#   --token_sample_size=512 \
#   --video_sample_stride=2 \
#   --video_sample_n_frames=81 \
#   --train_batch_size=1 \
#   --video_repeat=1 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=8 \
#   --num_train_epochs=100 \
#   --checkpointing_steps=50 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="output_dir" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=1 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --training_with_video_token_length \
#   --enable_bucket \
#   --uniform_sampling \
#   --boundary_type="low" \
#   --train_mode="i2v" \
#   --low_vram 
