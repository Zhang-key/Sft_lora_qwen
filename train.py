import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from dataclasses import dataclass, field
import torch

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_SEQ_LENGTH = 4096

@dataclass
class FinetuneArguments:
    model_path: str = field(default="/home/xuafeng/Qwen3-1.7B")
    train_file: str = field(default="./Final_dataset_merge/train.jsonl")
    output: str = field(default="./output/Qwen3-1.7B_1_epoch")
    merged_model: str = field(default="./output/Qwen3-1.7B_sft_test")
    checkpoint: str = field(default="")

def process_func(example, tokenizer):
    instruction = tokenizer(
        "\n".join([
            "<|im_start|>system",
            example["Sys_prompt"] + "\n<|im_end|>",
            "<|im_start|>user\n" + example["User_prompt"] + "<|im_end|>\n"
        ]).strip(),
        add_special_tokens=False
    )
    response = tokenizer(
        "<|im_start|>assistant\n" + example["Response"] + "<|im_end|>\n",
        add_special_tokens=False
    )
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        attention_mask = attention_mask[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    parser = HfArgumentParser((FinetuneArguments, TrainingArguments))
    finetune_args, training_args = parser.parse_args_into_dataclasses()

    training_args.per_device_train_batch_size = getattr(training_args, "per_device_train_batch_size", 4)
    training_args.gradient_accumulation_steps = getattr(training_args, "gradient_accumulation_steps", 8)
    training_args.logging_steps = getattr(training_args, "logging_steps", 32)
    training_args.num_train_epochs = getattr(training_args, "num_train_epochs", 1)
    training_args.gradient_checkpointing = True
    training_args.save_steps = getattr(training_args, "save_steps", 32)
    training_args.learning_rate = getattr(training_args, "learning_rate", 1e-4)
    training_args.save_on_each_node = True
    training_args.fp16 = True

    # Load dataset.
    df = pd.read_json(finetune_args.train_file, lines=True)
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(finetune_args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_id = ds.map(lambda x: process_func(x, tokenizer), remove_columns=ds.column_names)

    # Load model.
    model = AutoModelForCausalLM.from_pretrained(
        finetune_args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.half,
        device_map="auto"
    )

    # Lora settings.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    print("Start to merge Lora weights and the base model!!!")
    if finetune_args.checkpoint and os.path.exists(finetune_args.checkpoint):
        checkpoint = finetune_args.merge_checkpoint
        print(f"Using checkpoint: {checkpoint}")
    else:
        checkpoint_dir = finetune_args.output
        checkpoint = max(
            [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
            key=os.path.getmtime
        )
        print(f"Using checkpoint: {checkpoint}")

    base_model = AutoModelForCausalLM.from_pretrained(
        finetune_args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(base_model, checkpoint, device_map="cpu")
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(finetune_args.merged_model)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.save_pretrained(finetune_args.merged_model)

    print(f"Model is saved as: {finetune_args.merged_model}")

if __name__ == "__main__":
    main()


'''
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

'''


