import os
import json



# 线程限制（必须）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from paddleocr import PPChatOCRv4Doc


# LLM 配置
chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": os.getenv("MODEL_NAME", "qwen3-8b"),
    "base_url": os.getenv("MODEL_BASE_URL", "https://qianfan.baidubce.com/v2"),
    "api_type": "openai",
    "api_key": os.getenv("MODEL_API_KEY", ""),
}


TEMPLATE_MAP = {
    "passport": "passport.json",
    "application": "application form.json",
    "diploma": "diploma.json",
    "english": "English.json",
    "transcript": "transcript.json",

}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.getenv("INPUT_ROOT", os.path.join(BASE_DIR, "data"))
TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", os.path.join(BASE_DIR, "templates"))
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", os.path.join(BASE_DIR, "output"))



pipeline = PPChatOCRv4Doc(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec"
)

# 选择模板
def choose_template(input_file):
    filename = os.path.basename(input_file).lower()
    for keyword, template_name in TEMPLATE_MAP.items():
        if keyword in filename:
            print(f"[模板匹配] {filename} -> {template_name}")
            return os.path.join(TEMPLATE_DIR, template_name)
    return None

# 读取模板
def load_template(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 写入txt
def save_result_to_txt(template, field_meta, chat_data, input_file):
    person_id = os.path.basename(os.path.dirname(input_file))
    output_dir = os.path.join(OUTPUT_ROOT, person_id)
    os.makedirs(output_dir, exist_ok=True)

    input_filename = os.path.basename(input_file)
    output_filename = os.path.splitext(input_filename)[0] + ".txt"
    output_file = os.path.join(output_dir, output_filename)

    with open(output_file, "w", encoding="utf-8") as f:
        doc_type = template.get("doc_type", "")
        if doc_type:
            f.write(f"#{doc_type}\n")

        for item in field_meta:
            key = item.get("name", "")
            value = chat_data.get(key, "未找到")
            f.write(f"{key}: {value}\n")

    return output_file


# 从模板中解析字段
def parse_fields(template):
    raw_fields = template.get("fields", [])
    # 只用于输入统一成标准结构
    field_names = []
    field_meta = []
    
    if not raw_fields:
        return field_names, field_meta

    # 原始模板：["name", "date of birth"]
    if isinstance(raw_fields[0], str):
        for x in raw_fields:
            name = x.strip()
            if not name:
                continue

            field_names.append(name)
            field_meta.append({
                "name": name,
                "note": "",
                "section": ""
            })


        return field_names, field_meta

    # 增强模板：[{"name": "...", "section": "...", "note": "..."}]
    for item in raw_fields:
        name = item.get("name", "").strip()
        note = item.get("note", "").strip()
        section = item.get("section", "").strip()

        if not name:
            continue

        field_names.append(name)
        field_meta.append({
            "name": name,
            "note": note,
            "section": section
        })

    return field_names, field_meta




#把模板原信息转换成prompt（保持key_list干净，把section/note 作为额外提示传给模型）
def build_text_prompt_from_template(template, field_meta):
    doc_type = template.get("doc_type", "").strip()

    lines = []
    if doc_type:
        lines.append(f"Document type: {doc_type}")

    lines.append("Please extract the following fields from the document.")
    lines.append("Use the original field names exactly as provided in key_list as output keys.")
    lines.append("If a field cannot be found, return an empty string for that field.")
    lines.append("The following field hints may help locate the information:")

    for item in field_meta:
        name = item.get("name", "").strip()
        section = item.get("section", "").strip()
        note = item.get("note", "").strip()

        hint_parts = []
        if section:
            hint_parts.append(f"section={section}")
        if note:
            hint_parts.append(f"note={note}")

        if hint_parts:
            lines.append(f"- {name}: " + "; ".join(hint_parts))
        else:
            lines.append(f"- {name}")

    return "\n".join(lines)



# 主函数
def extract_fields(input_path, field_list,text_task_description=None, text_rules_str=None):

    

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
        text_task_description=text_task_description,
        text_rules_str=text_rules_str,
        text_output_format="JSON object of key-value pairs."

    )

    return chat_result


# 处理单个文件
def process_one_file(input_file):
    print(f"[开始] 正在处理: {input_file}")

    if not os.path.exists(input_file):
        print(f"[不存在] {input_file}")
        return

    template_file = choose_template(input_file)
    if template_file is None:
        print(f"[跳过] 未匹配模板: {input_file}")
        return
    print(f"[模板] 使用模板: {template_file}")
    try:
        template = load_template(template_file)
        fields, field_meta = parse_fields(template)
        print(f"[字段] 共解析 {len(fields)} 个字段")
        print(f"[字段列表] {fields}")


        text_task_description = build_text_prompt_from_template(template, field_meta)
        text_rules_str = (
            "Extract the value for each key from the document. "
            "Use exactly the original field names from key_list as output keys. "
            "Do not rename keys. "
            "If a field is not found, return an empty string."
        )

        print(f"[任务描述]\n{text_task_description}")

        result = extract_fields(
            input_file, 
            fields,
            text_task_description=text_task_description, 
            text_rules_str=text_rules_str
            )
        print(f"[原始结果类型] {type(result).__name__}")
        print("RESULT:", result)
        

        chat_data = result.get("chat_res", {}) if isinstance(result, dict) else {}
        print(f"[LLM返回原始keys] {list(chat_data.keys())}")
        print(f"[抽取结果] 提取到 {len(chat_data)} 个字段")


        output_file = save_result_to_txt(template, field_meta, chat_data, input_file)
        print(f"[完成] {input_file} -> {output_file}")

    except Exception as e:
        print(f"[失败] {input_file}: {e}")


def process_all_files(input_root):
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".pdf"):
                input_file = os.path.join(root, file)
                process_one_file(input_file)



# 提取ocr文本
def flatten_text(obj, texts):
    if obj is None:
        return

    if isinstance(obj, str):
        s = obj.strip()
        if s:
            texts.append(s)
        return

    if isinstance(obj, list):
        for item in obj:
            flatten_text(item, texts)
        return

    if isinstance(obj, dict):
        for key in ["text", "rec_text", "label", "words"]:
            if key in obj and isinstance(obj[key], str):
                s = obj[key].strip()
                if s:
                    texts.append(s)

        for v in obj.values():
            flatten_text(v, texts)

# 保存ocr.txt
def save_ocr_text_to_txt(input_file, visual_predict_res):
    person_id = os.path.basename(os.path.dirname(input_file))
    output_dir = os.path.join(OUTPUT_ROOT, person_id)
    os.makedirs(output_dir, exist_ok=True)

    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{input_filename}_ocr.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("#OCR_TEXT\n\n")
        f.write(f"source_file: {input_file}\n")
        f.write(f"page_count: {len(visual_predict_res)}\n\n")

        for page_idx, res in enumerate(visual_predict_res, start=1):
            f.write(f"## Page {page_idx}\n")

            texts = []
            if isinstance(res, dict):
                visual_info = res.get("visual_info", res)
                flatten_text(visual_info, texts)
            else:
                flatten_text(res, texts)

            seen = set()
            cleaned = []
            for t in texts:
                if t not in seen:
                    seen.add(t)
                    cleaned.append(t)

            if cleaned:
                for line in cleaned:
                    f.write(line + "\n")
            else:
                f.write("[无可提取文本]\n")

            f.write("\n")

    return output_file



def process_one_file_ocr_only(input_file):
    print(f"[开始] 正在处理: {input_file}")

    if not os.path.exists(input_file):
        print(f"[不存在] {input_file}")
        return

    try:
        visual_predict_res = extract_ocr_only(input_file)
        print(f"[OCR] 页数: {len(visual_predict_res)}")

        ocr_txt_file = save_ocr_text_to_txt(input_file, visual_predict_res)
        print(f"[OCR TXT] 已保存: {ocr_txt_file}")

    except Exception as e:
        print(f"[失败] 文件: {input_file}")
        print(f"[异常类型] {type(e).__name__}")
        print(f"[异常信息] {e}")



# 只跑ocr，用于测试
def extract_ocr_only(input_path):
    visual_predict_res = pipeline.visual_predict(
        input=input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
    )
    return visual_predict_res



# 执行部分
if __name__ == "__main__":
    input_file = os.getenv("INPUT_FILE", os.path.join(INPUT_ROOT, "sample_application.pdf"))
    process_one_file(input_file)
