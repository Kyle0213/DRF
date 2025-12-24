import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
import csv
import bitsandbytes

gpu_id = 5
with torch.cuda.device(f"cuda:{gpu_id}"):  # 指定 GPU
    torch.cuda.empty_cache()
# -------------------------------
# 模型加载及 QLoRA 相关配置
# -------------------------------
model_name = "/data/yyr/cot/model/llama"  
save = "llama_500_2e"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 使用eos作为pad
# 设置左侧填充，生成任务一般要求有效 token 靠右对齐
tokenizer.padding_side = "right"
lrate=2e-5
num_epochs = 10
# 使用更大的最大长度，避免 prompt 过长被截断
max_length = 1024
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
ch2l = load_dataset("json", data_files="sqa_500.json")

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": f"cuda:{gpu_id}"},
    use_cache=False  # 显式关闭缓存
)


# 针对 k-bit 训练进行预处理，冻结大部分参数
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,  # 增大秩
    lora_alpha=32,  # alpha=2*r
    target_modules=["q_proj","k_proj","v_proj","o_proj"],  # 覆盖全部注意力模块
    lora_dropout=0.1,  # 降低dropout
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def combine_and_mask(prompt, target_text):
    # 编码 prompt 和 target
    prompt_enc = tokenizer(prompt, add_special_tokens=False, truncation=False)
    target_enc = tokenizer(target_text, add_special_tokens=False, truncation=False)
    
    prompt_ids = prompt_enc.input_ids
    target_ids = target_enc.input_ids
    original_prompt_len = len(prompt_ids)
    original_target_len = len(target_ids)
    
    # 拼接 prompt + target + EOS
    full_input_ids = prompt_ids + target_ids
    if tokenizer.eos_token_id is not None:
        full_input_ids.append(tokenizer.eos_token_id)
    
    current_length = len(full_input_ids)
    
    # --------------- 核心修改：右侧截断逻辑 ---------------
    # 如果总长度超过限制，优先截断左侧的 prompt
    if current_length > max_length:
        # 计算需要保留的 target 长度（至少保留完整 target）
        keep_target_len = min(original_target_len + 1, max_length)  # +1 for EOS
        # 剩余长度分配给 prompt
        keep_prompt_len = max(max_length - keep_target_len, 0)
        # 截断左侧的 prompt，保留右侧的 prompt 和完整的 target
        truncated_prompt = prompt_ids[-keep_prompt_len:] if keep_prompt_len > 0 else []
        full_input_ids = truncated_prompt + target_ids + [tokenizer.eos_token_id]
        current_length = len(full_input_ids)
        # 更新截断后的 prompt 长度
        original_prompt_len = len(truncated_prompt)
    
    # 计算填充长度
    pad_length = max(max_length - current_length, 0)
    
    # --------------- 右侧填充 ---------------
    padded_input_ids = full_input_ids + [tokenizer.pad_token_id] * pad_length
    attention_mask = [1] * current_length + [0] * pad_length
    
    # 确保最终长度正确
    padded_input_ids = padded_input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    
    # --------------- 动态计算 target 起始位置 ---------------
    # target_start_idx = 截断后的 prompt 长度
    target_start_idx = original_prompt_len
    
    # 初始化 labels（全为 -100）
    labels = [-100] * len(padded_input_ids)
    
    # 设置有效标签（右移一位）
    for i in range(target_start_idx, len(padded_input_ids)-1):
        next_token = padded_input_ids[i+1]
        # 仅当下一 token 非填充时设置 label
        if next_token != tokenizer.pad_token_id:
            labels[i] = next_token
    
    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }

# -------------------------------
# 数据预处理与任务设计
# -------------------------------
def preprocess_function1(examples):
    """
    任务1：输入为 context 和 hypothesis，输出为 thought 和 label
    """

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for question, q2a_Thought,q2a_Answer in zip(
        examples['question'], examples['q2a_[Thought]'],examples['q2a_[Answer]']
    ):
        prompt = f"""
        Please play the role of a Question Answering data builder. Now I will give an example, including [Qustion]. 
    [Example]
        [Question]: {question}
    Now you need to play the role of a Question Answering data builder, based on the above content, provide how you arrived at the corresponding answer and thought process.
    You need to first provide your thought process before giving the answer.
    Please give me your answer as the format as:
        [Thought]:
        [Answer]:(You could only reply "True" or "False" here)
    Please reply in less than 200 words.
    """
        # condition_text = f"context: {context}  hypothesis: {hypothesis} "
        target_text = f"""
        [Thought]: {q2a_Thought} 
        [Answer]: {q2a_Answer}
        """
        # print(context, label, thought, hypothesis)
        tokenized = combine_and_mask(prompt, target_text)
        input_ids_list.append(tokenized['input_ids'])
        attention_mask_list.append(tokenized['attention_mask'])
        labels_list.append(tokenized['labels'])
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }
    
    
def preprocess_function2(examples):
    """
    任务1：输入为 context 和 hypothesis，输出为 thought 和 label
    """

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for question, anwser,reason,qa2s_Thought, qa2s_Strategy, qa2s_KeyTerm in zip(
        examples['question'], examples['q2a_[Answer]'],examples['q2a_[Thought]'], examples['qa2s_[Thought]'],examples['qa2s_[Strategy]'],examples['qa2s_[KeyTerm]']    
        ):
        prompt = f"""
    Please play the role of a Question Answering data builder. Now I will give an [Example], including [Qustion], [Answer], [Reason]. 
    The question's construction strategy and corresponding example include the following：
        Physical: Can human nails carvea statue out of quartz?
        Biological: Is a platypus immune from cholera?
        Historical: Were mollusks an ingredient in the color purple?
        Temporal: Did the 40th president of the United States forward lolcats to his friends?
        Definition: Are quadrupeds represented on Chinese calendar?
        Culturall: Would a compass attuned to Earth’s magnetic field be a bad gift for a Christmas elf?
        Religious: Was Hillary Clinton’s deputy chief of staff in 2009 baptised?
        Entertainment: Would Garfield enjoy a trip to Italy?
        Sports: Can Larry King’s ex-wives form a water polo team?
    [Example]
        [Question]: {question}
        [Answer]: {anwser}
        [Reason]: {reason}
    You need to tell me which strategy the construction method for this question belongs to, and what the [KeyTerm] correspond to.
    [KeyTerm] is the core concept of this question construction.
    Provide how you arrived at the corresponding [KeyTerm] and thought process.
    Please give me your answer as the format as:
        [Thought]:
        [Strategy]:
        [KeyTerm]:
    And do not reply any other parts.
    Please reply in less than 200 words.
    """
        # condition_text = f"context: {context}  hypothesis: {hypothesis} "
        target_text = f"""
        [Thought]:{qa2s_Thought}
        [Strategy]:{qa2s_Strategy}
        [KeyTerm]:{qa2s_KeyTerm}
        """
        # print(context, label, thought, hypothesis)
        tokenized = combine_and_mask(prompt, target_text)
        input_ids_list.append(tokenized['input_ids'])
        attention_mask_list.append(tokenized['attention_mask'])
        labels_list.append(tokenized['labels'])
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


def preprocess_function3(examples):
    """
    任务3：输入为 context 和 label，输出为 thought 和 hypothesis
    """

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for qa2s_Thought,qa2s_Strategy,qa2s_KeyTerm,as2q_Thought,as2q_Question in zip(
        examples['qa2s_[Thought]'],examples['qa2s_[Strategy]'],examples['qa2s_[KeyTerm]'],examples['as2q_[Thought]'],examples['as2q_[Question]']
    ):
        prompt = f"""
    Please play the role of a Question Answering data builder. Now I will give an example, including [Strategy], [Strategy_Reason] and [KeyTerm]. 
    [Strategy] means that the [Question] is based on the strategy created by [KeyTerm].
    [KeyTerm] represent the core concepts used in the construction process of the [Question].
    [Strategy_Reason] means the reason why the [Question] is judged as the [Strategy].
    The question's construction strategy and corresponding example include the following：
        Physical: Can human nails carvea statue out of quartz?
        Biological: Is a platypus immune from cholera?
        Historical: Were mollusks an ingredient in the color purple?
        Temporal: Did the 40th president of the United States forward lolcats to his friends?
        Definition: Are quadrupeds represented on Chinese calendar?
        Culturall: Would a compass attuned to Earth’s magnetic field be a bad gift for a Christmas elf?
        Religious: Was Hillary Clinton’s deputy chief of staff in 2009 baptised?
        Entertainment: Would Garfield enjoy a trip to Italy?
        Sports: Can Larry King’s ex-wives form a water polo team?
    [Example]
        [Strategy_Reason]: {qa2s_Thought}
        [Strategy]:{qa2s_Strategy}
        [KeyTerm]:{qa2s_KeyTerm}
    Now you need to play the role of a Question Answering data builder, based on the above content, provide the process of creating corresponding [Question].
    Please answer as the format as:
        [Thought]:
        [Question]:
    And do not reply any other parts.
    Please reply in less than 200 words.
    """
        # condition_text = f"context: {context}  label: {label} "
        target_text = f"""
        [Thought]:{as2q_Thought}
        [Question]:{as2q_Question}
        """
        # print(context, label, thought, hypothesis)
        tokenized = combine_and_mask(prompt, target_text)
        input_ids_list.append(tokenized['input_ids'])
        attention_mask_list.append(tokenized['attention_mask'])
        labels_list.append(tokenized['labels'])
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


# -------------------------------
# 数据集加载、预处理及拆分
# -------------------------------
print(f"原始数据集大小: {len(ch2l['train'])}")
columns_to_remove_ch2l = ch2l["train"].column_names


tokenized1 = ch2l.map(preprocess_function1, batched=True, remove_columns=columns_to_remove_ch2l, num_proc=8)
tokenized2 = ch2l.map(preprocess_function2, batched=True, remove_columns=columns_to_remove_ch2l, num_proc=8)
tokenized3 = ch2l.map(preprocess_function3, batched=True, remove_columns=columns_to_remove_ch2l, num_proc=8)


def split_train_validation(dataset):
    split = dataset["train"].train_test_split(test_size=0.1)
    return split["train"], split["test"]

train1, validation1 = split_train_validation(tokenized1)
train2, validation2 = split_train_validation(tokenized2)
train3, validation3 = split_train_validation(tokenized3)

train_dataloader1 = DataLoader(train1, batch_size=4, shuffle=False, collate_fn=default_data_collator)
train_dataloader2 = DataLoader(train2, batch_size=4, shuffle=False, collate_fn=default_data_collator)
train_dataloader3 = DataLoader(train3, batch_size=4, shuffle=False, collate_fn=default_data_collator)

eval_dataloader1 = DataLoader(validation1, batch_size=4, collate_fn=default_data_collator)
eval_dataloader2 = DataLoader(validation1, batch_size=4, collate_fn=default_data_collator)
eval_dataloader3 = DataLoader(validation3, batch_size=4, collate_fn=default_data_collator)

optimizer = AdamW(model.parameters(), lr=lrate)
loss_fn = CrossEntropyLoss(ignore_index=-100)


# -------------------------------
# 训练与评估函数
# -------------------------------

scaler = GradScaler()  # 新增 GradScaler
from tqdm.auto import tqdm

def train_epoch(model, dataloader1, dataloader2,dataloader3, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    with tqdm(
        zip(dataloader1, dataloader2,dataloader3), 
        total=len(dataloader1), 
        desc="Training",
        dynamic_ncols=True
    ) as pbar:
        for step, (batch1,batch2, batch3) in enumerate(pbar):
            # 数据移动到设备
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            batch2 = {k: v.to(device) for k, v in batch2.items()}
            batch3 = {k: v.to(device) for k, v in batch3.items()}

            optimizer.zero_grad()

            # 混合精度前向计算
            with autocast(device_type="cuda"):
                outputs1 = model(**batch1)
                loss1 = loss_fn(outputs1.logits.view(-1, outputs1.logits.size(-1)), batch1['labels'].view(-1))
                
                outputs2 = model(**batch2)
                loss2 = loss_fn(outputs2.logits.view(-1, outputs2.logits.size(-1)), batch2['labels'].view(-1))

                outputs3 = model(**batch3)
                loss3 = loss_fn(outputs3.logits.view(-1, outputs3.logits.size(-1)), batch3['labels'].view(-1))
                combined_loss = 0.4*loss1+0.3*loss2+0.3*loss3

            # 反向传播与优化
            scaler.scale(combined_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += combined_loss.item()

            # 更新进度条
            pbar.set_postfix(
                loss1=loss1.item(),
                loss2=loss2.item(),
                loss3=loss3.item(),
                total_loss=combined_loss.item(),
                refresh=False
            )

    return total_loss / len(dataloader1)

def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0

    with tqdm(
        dataloader, 
        desc="Evaluating", 
        dynamic_ncols=True
    ) as pbar:
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                with autocast(device_type="cuda"):
                    outputs = model(**batch)
                    loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), batch['labels'].view(-1))

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), refresh=False)

    return total_loss / len(dataloader)

# -------------------------------
# 训练主循环
# -------------------------------

train_losses = []
eval_losses1 = []
eval_losses2 = []
eval_losses3 = []
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss = train_epoch(model, train_dataloader1, train_dataloader2, train_dataloader3, optimizer, loss_fn, device)
    train_losses.append(train_loss)
    save_path = f"./{save}/lora_model_epoch_{epoch+1}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    lora_config.save_pretrained(save_path)
    # eval_loss1 = evaluate(model, eval_dataloader1, device,loss_fn)
    # eval_loss2 = evaluate(model, eval_dataloader2, device,loss_fn)
    # eval_loss3 = evaluate(model, eval_dataloader3, device,loss_fn)
    # eval_losses1.append(eval_loss1)
    # eval_losses2.append(eval_loss1)
    # eval_losses3.append(eval_loss3)

    print(f"Train Loss: {train_loss:.4f}")
    # print(f"Eval Loss Dataset 1: {eval_loss1:.4f}")
    # print(f"Eval Loss Dataset 2: {eval_loss2:.4f}")
    # print(f"Eval Loss Dataset 3: {eval_loss3:.4f}")
    torch.cuda.empty_cache()
with open("./thought/losses.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "eval_loss1",  "eval_loss2","eval_loss3"])
    for epoch_idx, (tl, el1,  el3) in enumerate(zip(train_losses, eval_losses1, eval_losses3), start=1):
         writer.writerow([epoch_idx, tl, el1, el3])
