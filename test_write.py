import os

# ===== 模拟 LLM 返回 =====
result = {
    'chat_res': {
        '驾驶室准乘人数': '2人',
        '车辆型号': 'DHJ3554664524',
        '发动机号码': 'QWEQE25235243242'
    }
}

# ===== 需要的字段 =====
fields = [
    "驾驶室准乘人数",
    "车辆型号",
    "发动机号码",
]

# ===== 正确取数据 =====
chat_data = result.get("chat_res", {})

print("chat_data:", chat_data)
print("chat_data keys:", list(chat_data.keys()))

# ===== 写入文件 =====
output_file = os.getenv("OUTPUT_FILE", os.path.join(os.path.dirname(__file__), "output", "sample.txt"))
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    for key in fields:
        value = chat_data.get(key, "未找到")
        print(f"写入 -> {key}: {value}")
        f.write(f"{key}: {value}\n")

print("\n写入完成！")
