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
    device_map={"": 2}  # 修改数字选择GPU设备
)
save_path='./sqa_rev_qwen_2e.json'
tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("./sqa_test.json", "r", encoding="utf-8") as infile:
    dataset = json.load(infile)
# 初始化文本生成管道
 
 
def process_generation(text):
    """
    后处理生成文本，提取关键信息
    返回格式：(analysis, label)
    """
    # 定义正则表达式模式
    pattern = r'\[(Thought|Answer)\]:\s*(.*?)(?=\n\[|$)'
    
    # 使用re.findall提取所有匹配的内容
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 创建一个字典来存储提取的结果
    result = {}
    for match in matches:
        key = match[0]  # 提取的标签名
        value = match[1].strip()  # 提取的内容并去除首尾空格
        result[key] = value
    
    return result

def generate_response(question):
    """生成英文prompt并获取模型响应"""
    prompt = f"""
        Please play the role of a Question Answering data builder. Now I will give an example, including [Qustion]. 
    [Example]
        [Question]: {question}
    Now you need to play the role of a Question Answering data builder, based on the above content， provide how you arrived at the corresponding answer and thought process.
    You need to first provide your thought process before giving the [Answer].
    Please give me your reply as the format with two parts:
        [Thought]:
        [Answer]: (You could only reply "True" or "False" here)
    You must give your reply with [Answer].
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant for SQA"},
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
    question = item.get("question", "")
    
    # 获取模型输出
    # response = generate_response(question, a,b,c,d,e)
    response,extracted_answer = generate_response(question)
    analysis = extracted_answer.get("Thought", "")
    pred_ans = extracted_answer.get("Answer", "")
    # 构建输出结构
    processed_item = {
        "question": question,
        "original_ans": item.get("gold_answer", ""),  # 原始标注
        "predicted_ans": pred_ans,            # 模型预测
        "analysis": analysis,                      # 分析过程
        "response":response
    }
    processed_data.append(processed_item)
    # 保存结果文件
    with open(f"{save_path}", "w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)
