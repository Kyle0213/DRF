import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm.auto import tqdm
import torch
# 加载8bit量化模型到指定GPU
model_name='/data/yyr/cot/RevThink/checkpoints/qwen_SQA_100'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 5}  # 修改数字选择GPU设备
)
save_path='./ARC_qwen_rev.json'
tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("./ARC_dev.json", "r", encoding="utf-8") as infile:
    dataset = json.load(infile)
# 初始化文本生成管道
 
def process_generation(text):
    """
    后处理生成文本，提取关键信息
    返回格式：(analysis, label)
    """
    # 定义正则表达式模式
    pattern = r'\[(Thought|Choice)\]:\s*(.*?)(?=\n\[|$)'
    
    # 使用re.findall提取所有匹配的内容
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 创建一个字典来存储提取的结果
    result = {}
    for match in matches:
        key = match[0]  # 提取的标签名
        value = match[1].strip()  # 提取的内容并去除首尾空格
        result[key] = value
    
    return result

def generate_response(question, choice_A, choice_B, choice_C, choice_D):
    """生成英文prompt并获取模型响应"""
    prompt = f"""
    Please play the role of a Question Answering data builder. Now I will give an example, including [Qustion]. 
    [Example]
        [Question]: {question}
        [Choices]:[A]:{choice_A} [B]:{choice_B} [C]:{choice_C} [D]:{choice_D}
    Please reply only (A/B/C/D/E)
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant for natural language inference."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # return(response)
    return response,process_generation(response)

# 主处理流程


processed_data = []
for item in tqdm(dataset, desc="处理进度"):
    question = item.get("question", "").get("stem")
    choices = item.get("question").get("choices")

    a = choices[0].get("text")
    b = choices[1].get("text")
    c = choices[2].get("text")
    d = choices[3].get("text")
    
    # 获取模型输出
    # response = generate_response(question, a,b,c,d,e)
    reply,extracted_answer = generate_response(question, a,b,c,d)
    analysis = extracted_answer.get("Thought", "")
    pred_ans = extracted_answer.get("Choice", "")
    # 构建输出结构
    processed_item = {
        "question": question,
        "choice_A": a,
        "choice_B": b,
        "choice_C": c,
        "choice_D": d,
        "original_ans": item.get("answerKey", ""),  # 原始标注
        "predicted_ans": pred_ans,            # 模型预测
        "analysis": analysis,                      # 分析过程
        "response":reply
    }
    processed_data.append(processed_item)
    # 保存结果文件
    with open(f"{save_path}", "w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)
