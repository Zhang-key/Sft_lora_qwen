# SFT_lora_qwen

Training settings are listed below and can be customized as needed.

python train.py \
  --model_path /home/xuafeng/Qwen3-1.7B \
  --train_file ./Final_dataset_merge/train.jsonl \
  --output ./output/Qwen3-1.7B_1_epoch \
  --checkpoint ./output/Qwen3-1.7B_1_epoch/checkpoint-64 \
  --merged_model /home/xuafeng/Qwen3-1.7B_checkpoint_64 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --logging_steps 64\
  --save_steps 64\
  --learning_rate 5e-5
