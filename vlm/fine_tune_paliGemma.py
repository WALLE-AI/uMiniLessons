import os
from datasets import load_dataset
import loguru
from transformers import PaliGemmaProcessor
from transformers import PaliGemmaForConditionalGeneration
import torch
device = "cuda"
from transformers import  PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments

from transformers import Trainer
from swanlab.integration.huggingface import SwanLabCallback
from PIL import Image


swanlab_callback = SwanLabCallback(
    project="paligemema",
    experiment_name="paligemema-3b",
    description="paligemama loar fine tune",
    config={
        "model": "paligemma2-3b-pt-224",
        "model_dir": "/home/dataset1/gaojing/models/paligemma2-3b-pt-224",
        "dataset": "vqav2-small-sample",
    },
)

##设置wand环境变量
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="paligemema"


lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

def load_datasets():
    ds = load_dataset('parquet', data_files= '/home/dataset1/gaojing/llm/uMiniLessons/datasets/vqav2-small/validation-00000-of-00007.parquet',split='train')
    split_ds = ds.train_test_split(test_size=0.2) # we'll use a very small split for demo
    train_ds= split_ds["train"]
    loguru.logger.info(f"train example:{train_ds[0]},train size:{len(train_ds)}")
    return train_ds

def init_model(model_id):
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    for param in model.vision_tower.parameters():
        param.requires_grad = True

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
        
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto")#, quantization_config=bnb_config)
    model = get_peft_model(model, lora_config)
    loguru.logger.info(f"train paremeter:{model.print_trainable_parameters()}")
    return model,processor



args=TrainingArguments(
            num_train_epochs=2,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf", # you can use paged optimizers like paged_adamw_8bit for QLoRA
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=1,
            output_dir="paligemma_vqav2",
            bf16=True,
            report_to="none",
            dataloader_pin_memory=False
        )

def execute_train():
    model_id = "/home/dataset1/gaojing/models/paligemma2-3b-pt-224"
    train_ds = load_datasets()
    model,processor = init_model(model_id)
    # image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
    def build_datasets(examples):
        image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
        texts = ["<image>" + example["question"] for example in examples]
        labels= [example['multiple_choice_answer'] for example in examples]
        ##resize((128,128),Image.Resampling.LANCZOS
        images = [example["image"].convert("RGB") for example in examples]
        loguru.logger.info(f"text:{texts},lables:{labels},images:{images}")
        tokens = processor(text=texts, images=images, suffix=labels,
                            return_tensors="pt", padding="longest")

        tokens = tokens.to(model.dtype).to(device)
        loguru.logger.info(f"token shape:{tokens['pixel_values'].shape,tokens['input_ids'].shape,tokens['labels'].shape}")
        return tokens
    trainer = Trainer(
        model=model,
        train_dataset=train_ds ,
        data_collator=build_datasets,
        args=args,
        # callbacks=[swanlab_callback]
        )
    trainer.train()
    
    
if __name__ == "__main__":
    loguru.logger.info(f"start paligemma fine tune")
    execute_train()
    
    


