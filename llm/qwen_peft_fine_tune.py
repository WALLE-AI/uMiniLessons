import json
import loguru
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq,AutoTokenizer
import torch
from swanlab.integration.huggingface import SwanLabCallback
import swanlab
from datasets import Dataset

'''
业务上面微调一般执行如下步骤：
1、先收集业务的数据，通过prompt测试，观察覆盖的精度指标，比如acc、recall等指标（开源模型和商业模型）
2、如果精度无法满足业务精度要求，分析当前的业务场景task主要模型什么技能：
    如何是知识注入，采用sft几乎没有什么作用，如何是模式输出prompt无法解决固定prompt无法保证输出格式，这个使用使用也因为5k的数据直接lora引导模型正确输出
    简答的NLP任务，比如NER text classification IE问题等等可以通过微调来解决acc精度问题，这类问题还是充分发挥模型few-show和ICL能力，一种是prompt解决 一种prompt+微调（lora）
3、如果当前业务场景是复杂领域问题，比如知识注入、理解和综合性推理，这个时候仅仅SFT无法满足业务训练任务，只能通过CPT(持续预训练)，如果推理能力提升或者逻辑能力提升要采用DPO训练或者PPO ReFT来训练

如下基于qwen2.5模型进行 ner任务和文本分类任务lora问题，其实大家也可以基于transformers来进行训练代码集成开发，目前主流的训练平台大部分是基于transformers开发的。比如llama-factory medicalGPT等等
'''


pretrained_model_path = "/root/autodl-tmp/models/Qwen2-0.5B-Instruct" 
##注意不同的模型微调的prompt template不一样 要主要specal_tokens
prompt_template_input = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'''


def load_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False, trust_remote_code=True)    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    return model,tokenizer

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)


##也可以使用wandb来写这个训练观测的配置问题
swanlab_callback = SwanLabCallback(
    project="Qwen2-NER-fintune",
    experiment_name="Qwen2-0.5B-Instruct",
    description="使用通义千问Qwen2-1.5B-Instruct模型在NER数据集上微调，实现关键实体识别任务。",
    config={
        "model": "Qwen2-0.5B-Instruct",
        "model_dir": "/root/autodl-tmp/models/Qwen2-0.5B-Instruct",
        "dataset": "test_ner_sft",
    },
)


def build_train_sft_datasets(tokenizer):
    dataset_json_file_path = "data/race_data/train_sft_dataset.json"
    total_df = pd.read_json(dataset_json_file_path, lines=True)
    train_df = total_df[int(len(total_df) * 0.1):]
    train_ds = Dataset.from_pandas(total_df)
    train_dataset = train_ds.map(process_func, with_arg = {"tokenizer":tokenizer},remove_columns=train_ds.column_names)
    loguru.logger.info(f"train_dataset size {len(train_dataset)}")
    return train_dataset,total_df
    
    
def process_func(example,tokenizer):
    """
    将数据集进行预处理
    """

    MAX_LENGTH = 1024 
    input_ids, attention_mask, labels = [], [], []
    system_prompt = example['instruction']
    prompts = prompt_template_input.format(system_prompt=system_prompt,input=example['input'])
    instruction = tokenizer(
        prompts,
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def model_predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def eval_dataset(total_df,model,tokenizer):
    test_df = total_df[:int(len(total_df) * 0.1)].sample(n=20)

    test_text_list = []
    for index, row in test_df.iterrows():
        instruction = row['instruction']
        input_value = row['input']
    
        messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
        ]

        response = model_predict(messages, model, tokenizer)
        messages.append({"role": "assistant", "content": f"{response}"})
        result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
        test_text_list.append(swanlab.Text(result_text, caption=response))
    
    swanlab.log({"Prediction": test_text_list})
    swanlab.finish()




def train():
    out_put_dir = "/root/autodl-tmp/models/checkpoint_dir/qwen"
    model,tokenizer = load_model_tokenizer()
    model = get_peft_model(model, config)
    train_dataset,total_df = build_train_sft_datasets()
    args = TrainingArguments(
        output_dir=out_put_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
    )
    trainer.train()
    ##随机抽取20条测试
    eval_dataset(total_df,model)
    
    
if __name__ == "__main__":
    # train()
    pass


