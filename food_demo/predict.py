import torch
import pandas as pd
from tqdm import tqdm  # 用于显示进度条
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def predict(messages, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 模型路径设置
base_model_path = "../../glm-4-9b-chat/"  # 原始模型路径
lora_model_path = "../../LLaMA-Factory-main3/chatglm4/saves/train8400_data1_20250322_1511_json回答/checkpoint-7860"  # 微调模型路径

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, model_id=lora_model_path)

# 输入/输出路径
input_file = "test_data.csv"
output_file = "output_data_8400.csv"
error_log_file = "error_log.txt"

# 读取 CSV 文件
df = pd.read_csv(input_file)

# 结果缓存列表
results = []

# 开始处理并显示进度条
with open(error_log_file, "w") as error_log:
    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        user_input = row['文本']
        try:
            # 构建对话
            messages = [
                {"role": "system", "content": "对下面食品舆情数据进行处理："},
                {"role": "user", "content": user_input}
            ]

            # 获取模型输出
            response = predict(messages, model, tokenizer)

            # 解析模型输出
            if "食品安全事件" in response:
                try:
                    output_data = eval(response)  # 将字符串转为字典
                    label = output_data.get('标签', '')
                    region = output_data.get('地域', '')
                    food_types = ', '.join(output_data.get('食品种类', []))
                    organizations = output_data.get('组织', '')
                    results.append([row['编号'], label, region, food_types, organizations, user_input])
                except Exception as e:
                    error_log.write(f"[解析错误] 编号: {row['编号']} | 错误: {str(e)} | 原始返回: {response}\n")
            else:
                results.append([row['编号'], "非食品安全事件", "", "", "", user_input])

        except Exception as e:
            error_log.write(f"[模型错误] 编号: {row['编号']} | 错误: {str(e)}\n")

# 写出结果
output_df = pd.DataFrame(results, columns=["编号", "标签", "地域", "食品种类", "组织", "文本"])
output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n处理完成 ✅\n输出文件: {output_file}\n错误日志: {error_log_file}")

