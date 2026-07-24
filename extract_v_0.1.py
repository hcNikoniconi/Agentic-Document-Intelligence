import os
import json
from paddleocr import PPChatOCRv4Doc


# 线程限制（必须）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# LLM 配置
chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": os.getenv("MODEL_NAME", "ernie-3.5-8k"),
    "base_url": os.getenv("MODEL_BASE_URL", "https://qianfan.baidubce.com/v2"),
    "api_type": "openai",
    "api_key": os.getenv("MODEL_API_KEY", ""),
}


# 读取模板
def load_template(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 从模板中解析字段
def parse_fields(template):
    raw_fields = template.get("fields", [])

    # 最简格式：["name", "date of birth"]
    if raw_fields and isinstance(raw_fields[0], str):
        return raw_fields, [{"name": x, "note": "", "section": ""} for x in raw_fields]

    # 扩展格式：[{"name":"name","note":"...","section":"2.1"}]
    field_names = []
    field_meta = []
    for item in raw_fields:
        name = item.get("name", "").strip()
        if not name:
            continue
        field_names.append(name)
        field_meta.append({
            "name": name,
            "note": item.get("note", "").strip(),
            "section": item.get("section", "").strip()
        })

    return field_names, field_meta


# 主函数
def extract_fields(input_path, field_list):

    pipeline = PPChatOCRv4Doc()

    # 1. OCR
    visual_predict_res = pipeline.visual_predict(
        input=input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
    )

    visual_info_list = []
    for res in visual_predict_res:
        visual_info_list.append(res["visual_info"])

    # 2. LLM 抽取
    chat_result = pipeline.chat(
        key_list=field_list,
        visual_info=visual_info_list,
        vector_info=None,
        mllm_predict_info=None,
        chat_bot_config=chat_bot_config,
        retriever_config=None,
    )

    return chat_result


# 执行部分
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.getenv("INPUT_FILE", os.path.join(base_dir, "data", "sample_passport.pdf"))
    template_file = os.getenv("TEMPLATE_FILE", os.path.join(base_dir, "templates", "passport.json"))

     # 读取模板
    template = load_template(template_file)
    fields, field_meta = parse_fields(template)

    # 执行抽取
    result = extract_fields(input_file, fields)

    print("RESULT:", result)

    chat_data = result.get("chat_res", {})
    print("CHAT_DATA:", chat_data)

    # 写入 txt
    person_id = os.path.basename(os.path.dirname(input_file))  # 取上一级目录名
    output_dir = os.path.join(os.getenv("OUTPUT_ROOT", os.path.join(base_dir, "output")), person_id)

    # 先创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出文件的命名
    input_filename = os.path.basename(input_file)
    output_filename = os.path.splitext(input_filename)[0] + ".txt"

    # 最终输出文件
    output_file = os.path.join(output_dir, output_filename)


    with open(output_file, "w", encoding="utf-8") as f:
        # 标题（可选）
        doc_type = template.get("doc_type", "")
        if doc_type:
            f.write(f"#{doc_type}\n")

        for item in field_meta:
            key = item.get("name", "")
            value = chat_data.get(key, "未找到")

            note = item.get("note", "")
            section = item.get("section", "")

            # 最简输出
            if not note and not section:
                f.write(f"{key}: {value}\n")
            else:
                line = f"{key}: {value}"
                if note:
                    line += f" | note: {note}"
                if section:
                    line += f" | section: {section}"
                f.write(line + "\n")

    print(f"\n结果已保存到: {output_file}")
