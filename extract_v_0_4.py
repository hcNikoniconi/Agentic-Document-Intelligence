import os
import json
from openai import OpenAI
import uuid



# 线程限制（必须）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from paddleocr import PPChatOCRv4Doc



# LLM 配置
LLM_CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "qwen3-8b"),
    "base_url": os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "api_key": os.getenv("MODEL_API_KEY", ""),
}


client = OpenAI(
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
)

TEMPLATE_MAP = {
    "passport": "passport.json",

    "application form": "application_form.json",
    "application": "application_form.json",

    "transcript": "transcript.json",

    "ielts": "english_language.json",
    "toefl": "english_language.json",
    "pte": "english_language.json",
    "duolingo": "english_language.json",
    "det": "english_language.json",

    
    "certificate": "diploma_certificate.json",
    "diploma": "diploma_certificate.json",
}



DOC_TYPE_ORDER = [
    "passport",
    "application_form",
    "transcript",
    "diploma_certificate",
    "english_language",
]


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
            print(f"[模板匹配] {filename} -> {template_name} (keyword={keyword})")
            return os.path.join(TEMPLATE_DIR, template_name)
    return None

# 读取模板
def load_template(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 写入txt
def save_result_to_txt(template, field_meta, chat_data, input_file, output_root=None):
    output_root = output_root or OUTPUT_ROOT
    os.makedirs(output_root, exist_ok=True)

    input_filename = os.path.basename(input_file)
    output_filename = os.path.splitext(input_filename)[0] + ".txt"
    output_file = os.path.join(output_root, output_filename)

    with open(output_file, "w", encoding="utf-8") as f:
        doc_type = template.get("doc_type", "")
        if doc_type:
            f.write(f"#{doc_type}\n")

        for item in field_meta:
            key = item.get("name", "")
            value = chat_data.get(key, "")
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


def build_json_skeleton(field_names):
    return {name: "" for name in field_names}

#把模板原信息转换成prompt（保持key_list干净，把section/note 作为额外提示传给模型）
def build_text_prompt_from_template(template, field_meta):
    doc_type = template.get("doc_type", "").strip()
    field_names = [item["name"] for item in field_meta if item.get("name", "").strip()]

    lines = []
    lines.append("Extract the value for each key from the document.")
    lines.append("Return JSON only.")
    lines.append("Do not output any reasoning.")
    lines.append("Do not output analysis.")
    lines.append("Do not output <think> tags.")
    lines.append("Do not explain.")
    lines.append("Use exactly the following field names as output keys.")
    lines.append("Do not rename keys.")
    lines.append("If a field is not found, return an empty string.")
    lines.append("Ignore instructions, notes, policy text, checklists, and explanatory content unless they directly contain the target field value.")
    lines.append("Prefer explicit values that appear nearest to the target field label.")
    lines.append("For table-like content, match each field with its most likely corresponding cell value.")
    lines.append("Do not copy values from neighboring fields.")
    lines.append("")
    lines.append("[任务描述]")

    if doc_type:
        lines.append(f"Document type: {doc_type}")

    lines.append("Please extract the following fields from the document.")
    lines.append("Use the original field names exactly as provided below as output keys.")
    lines.append("If a field cannot be found, return an empty string for that field.")
    lines.append("")
    lines.append("Fields:")

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

    lines.append("")
    lines.append("Return JSON only in the following format:")
    lines.append(
        json.dumps(
            {name: "" for name in field_names},
            ensure_ascii=False,
            indent=2
        ) 
    )

    return "\n".join(lines)


def visual_result_to_text(visual_predict_res):
    page_texts = []

    for page_idx, res in enumerate(visual_predict_res, start=1):
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

        page_text = "\n".join(cleaned) if cleaned else "[无可提取文本]"
        page_texts.append(f"## Page {page_idx}\n{page_text}")

    return "\n\n".join(page_texts)



def call_llm_with_text(document_text, instruction):
    prompt = f"""
        [Document Text]
        {document_text}

        [Instruction]
        {instruction}
        """

    response = client.chat.completions.create(
        model=LLM_CONFIG["model_name"],
        messages=[
            {
                "role": "system",
                "content": "You are an information extraction assistant. Return JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        extra_body={"enable_thinking": False}
    )

    content = response.choices[0].message.content.strip()
    return content

def parse_llm_json(content, field_meta):
    raw = (content or "").strip()

    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]

    if raw.endswith("```"):
        raw = raw[:-3]

    raw = raw.strip()

    field_names = [item.get("name", "") for item in field_meta if item.get("name", "").strip()]
    skeleton = {name: "" for name in field_names}

    if not raw:
        print("[JSON解析失败] LLM 返回为空")
        return skeleton

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for name in field_names:
                value = data.get(name, "")
                skeleton[name] = "" if value is None else str(value)
        else:
            print(f"[JSON解析失败] 解析结果不是 dict: {type(data).__name__}")
    except Exception as e:
        print(f"[JSON解析失败] {e}")
        print(f"[原始输出] {raw}")

    return skeleton




# main method
def extract_fields(input_path, template, field_meta):

    

    # 1. OCR
    visual_predict_res = pipeline.visual_predict(
        input=input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
    )

    # 2. OCR 结果转文本
    document_text = visual_result_to_text(visual_predict_res)

    # 3. 根据模板生成 instruction
    instruction = build_text_prompt_from_template(template, field_meta)

    # 4. document_text + instruction 直接喂给 LLM
    llm_raw_output = call_llm_with_text(document_text, instruction)

    return llm_raw_output


def process_one_file(input_file, output_root=None):
    result = extract_one_file(input_file)
    output_file = save_result_to_txt(
        template=result["template"],
        field_meta=result["field_meta"],
        chat_data=result["chat_data"],
        input_file=input_file,
        output_root=output_root,
    )
    print(f"[完成] {input_file} -> {output_file}")
    return output_file, result





def process_all_files(input_root):
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(".pdf"):
                input_file = os.path.join(root, file)
                try:
                    process_one_file(input_file)
                except Exception as e:
                    print(f"[失败] {input_file}: {e}")



def process_folder(input_dir, output_root=None):
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"不是有效文件夹: {input_dir}")

    output_root = output_root or OUTPUT_ROOT
    os.makedirs(output_root, exist_ok=True)

    results_by_type = {}
    logs = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue

            input_file = os.path.join(root, file)

            try:
                result = extract_one_file(input_file)
                doc_type = result["doc_type"]

                if doc_type not in results_by_type:
                    results_by_type[doc_type] = result
                    logs.append(f"[成功] {file} -> {doc_type}")
                else:
                    logs.append(f"[重复] {file} -> {doc_type}，已存在同类型文件，当前跳过")
            except Exception as e:
                logs.append(f"[失败] {file}: {e}")

    combined_output_path = os.path.join(
        output_root,
        f"{uuid.uuid4().hex}_combined_result.txt"
    )
    save_combined_result_to_txt(results_by_type, combined_output_path)

    return combined_output_path, results_by_type, logs



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



def build_preview_text(template, field_meta, chat_data):
    lines = []
    doc_type = template.get("doc_type", "")
    if doc_type:
        lines.append(f"#{doc_type}")

    for item in field_meta:
        key = item.get("name", "")
        value = chat_data.get(key, "")
        lines.append(f"{key}: {value}")

    return "\n".join(lines)




def extract_one_file(input_file):
    print(f"[开始] 正在处理: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")

    template_file = choose_template(input_file)
    if template_file is None:
        raise ValueError(f"未匹配到模板: {os.path.basename(input_file)}")

    template = load_template(template_file)
    fields, field_meta = parse_fields(template)

    print(f"[模板] 使用模板: {template_file}")
    print(f"[字段] 共解析 {len(fields)} 个字段")
    print(f"[字段列表] {fields}")

    llm_raw_output = extract_fields(
        input_path=input_file,
        template=template,
        field_meta=field_meta
    )

    print(f"[原始结果类型] {type(llm_raw_output).__name__}")
    print("\n================ LLM RAW OUTPUT ================\n")
    print(llm_raw_output)
    print("\n================ END OF OUTPUT ================\n")

    chat_data = parse_llm_json(llm_raw_output, field_meta)

    # 新增：判断是否几乎全空
    non_empty_count = sum(
        1 for v in chat_data.values()
        if str(v).strip()
    )

    if non_empty_count == 0:
        raise ValueError("模型未返回有效 JSON，或所有字段均为空")

    preview_text = build_preview_text(template, field_meta, chat_data)

    return {
        "doc_type": template.get("doc_type", "unknown"),
        "template_file": template_file,
        "template": template,
        "field_meta": field_meta,
        "chat_data": chat_data,
        "preview_text": preview_text,
        "input_file": input_file,
        "llm_raw_output": llm_raw_output,
    }



def save_combined_result_to_txt(results_by_type, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for doc_type in DOC_TYPE_ORDER:
            f.write(f"#{doc_type}\n")

            result = results_by_type.get(doc_type)
            if not result:
                f.write("未提供或未识别到该类文件\n\n")
                continue

            field_meta = result.get("field_meta", [])
            chat_data = result.get("chat_data", {})

            for item in field_meta:
                key = item.get("name", "")
                value = chat_data.get(key, "")
                f.write(f"{key}: {value}\n")

            f.write("\n")

    return output_file



# 执行部分
if __name__ == "__main__":
    input_file = os.getenv("INPUT_FILE", os.path.join(INPUT_ROOT, "sample_application.pdf"))
    process_one_file(input_file)
