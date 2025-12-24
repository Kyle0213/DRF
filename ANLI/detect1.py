import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm.auto import tqdm
import torch
# 加载8bit量化模型到指定GPU
model_name='/data/yyr/cot/model/mistral'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 2}  # 修改数字选择GPU设备
)
save_path='./ANLI_mis.json'
tokenizer = AutoTokenizer.from_pretrained(model_name)
with open("./R3test.json", "r", encoding="utf-8") as infile:
    dataset = json.load(infile)
# 初始化文本生成管道
 
def extract_answer(text):
    # 定义正则表达式模式
    pattern = r'\[(Thought|Label)\]:\s*(.*?)(?=\n\[|$)'
    
    # 使用re.findall提取所有匹配的内容
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 创建一个字典来存储提取的结果
    result = {}
    for match in matches:
        key = match[0]  # 提取的标签名
        value = match[1].strip()  # 提取的内容并去除首尾空格
        result[key] = value
    
    return result

def generate_response(context, hypothesis):
    """生成英文prompt并获取模型响应"""
    # prompt = f"""
    # Please act as an NLI data construction expert. Now I will give an example, including context, hypothesis.
    # Could you please tell me how you derived the correct label by chain of thought.
    # [Example]:
    #     [Context]: {context}
    #     [Hypothesis]: {hypothesis}
    # You should give me your anwser after your anlysis with format as:
    #     [Thought]:
    #     [Label]:(c or n or e) where: "e" means implication(we can infer from the context that the hypothesis is correct), "n" means neutral(we cannot infer the correctness of the hypothesis from the context), and "c" means contradiction(we can infer from the context that the hypothesis is incorrect).
    # Please reply in less than 200 words.
    # """
        prompt = f"""
    [Example]:
        [Context]: {context}
        [Hypothesis]: {hypothesis}
        Reply the relationship between the Context and Hypothesis.You can choose the relationship from (c or n or e) where: "e" means implication(we can infer from the context that the hypothesis is correct), "n" means neutral(we cannot infer the correctness of the hypothesis from the context), and "c" means contradiction(we can infer from the context that the hypothesis is incorrect).
        Only reply c or n or e.
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

    return response,extract_answer(response)

# 主处理流程


processed_data = []
for item in tqdm(dataset, desc="处理进度"):
    ctx = item.get("context", "")
    hypo = item.get("hypothesis", "")
    
    response,extracted_answer = generate_response(ctx, hypo)
    analysis = extracted_answer.get("Thought", "")
    pred_ans = extracted_answer.get("Label", "")
    
    # 构建输出结构
    processed_item = {
        "context": ctx,
        "hypothesis": hypo,
        "original_label": item.get("gold_label", ""),  # 原始标注
        "predicted_label": pred_ans,            # 模型预测
        "analysis": analysis,                # 分析过程
        "response": response
    }
    processed_data.append(processed_item)
    # 保存结果文件
    with open(f"{save_path}", "w", encoding="utf-8") as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)

