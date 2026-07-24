import os
import json
import re
from openai import OpenAI
import uuid
import time

from validator import save_validation_report_to_html


# 线程限制（必须）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from paddleocr import PPChatOCRv4Doc



# LLM 配置
LLM_CONFIG = {
    "model_name": os.getenv("MODEL_NAME", "qwen3-8b"),
    "base_url": os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
    "api_key": os.getenv("MODEL_API_KEY", "EMPTY"),
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


if "pipeline" not in globals():
    pipeline = PPChatOCRv4Doc(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        seal_text_detection_model_name="PP-OCRv4_server_seal_det",
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
    lines.append("If [OCR Visual Signals] says school stamp detected: Yes, set 'school stamp present' to 'Yes'.")
    lines.append("If seal recognized text is provided, use it as evidence for the presence of a school stamp.")
    lines.append("If no seal is detected and the document text does not clearly mention a stamp or seal, set 'school stamp present' to 'Unknown'.")
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



def get_result_json(obj):
    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "json"):
        value = obj.json
        if callable(value):
            value = value()
        return value

    return None

def debug_print_seal_structure(visual_predict_res, max_hits=80):
    """
    临时调试用：
    打印 PaddleOCR visual_predict_res 中和 seal / stamp / 印章相关的结构。
    不做判断，只看真实返回长什么样。
    """
    print("\n========== SEAL DEBUG START ==========")

    hit_count = 0

    def summarize_value(value):
        if value is None:
            return "None"

        if isinstance(value, dict):
            return f"dict keys={list(value.keys())[:20]}"

        if isinstance(value, list):
            return f"list len={len(value)}"

        if isinstance(value, tuple):
            return f"tuple len={len(value)}"

        try:
            # numpy array 等对象
            shape = getattr(value, "shape", None)
            if shape is not None:
                return f"{type(value).__name__} shape={shape}"
        except Exception:
            pass

        text = str(value)
        if len(text) > 300:
            text = text[:300] + " ...[truncated]"
        return f"{type(value).__name__}: {text}"

    def get_json_like(obj):
        if obj is None:
            return None

        if isinstance(obj, dict):
            return obj

        if hasattr(obj, "json"):
            try:
                value = obj.json
                if callable(value):
                    value = value()
                return value
            except Exception as e:
                print(f"[SEAL DEBUG] failed to read .json: {e}")

        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except Exception as e:
                print(f"[SEAL DEBUG] failed to read .to_dict(): {e}")

        return None

    def walk(obj, path="root"):
        nonlocal hit_count

        if hit_count >= max_hits:
            return

        obj_json = get_json_like(obj)
        if obj_json is not None and obj_json is not obj:
            obj = obj_json

        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                walk(item, f"{path}[{idx}]")
            return

        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            key_lower = str(key).lower()
            value_text = str(value).lower()

            interested = False

            # 重点关注这些 key
            if any(x in key_lower for x in [
                "seal",
                "stamp",
                "印章",
                "rec_texts",
                "rec_scores",
                "dt_polys",
                "rec_polys",
                "rec_boxes",
                "text_type",
                "block_label",
                "block_content",
            ]):
                interested = True

            # 也看 value 里是否直接出现 seal / stamp
            if any(x in value_text for x in [
                "seal",
                "stamp",
                "印章",
                "公章",
                "盖章",
            ]):
                interested = True

            if interested:
                hit_count += 1
                print(f"[SEAL DEBUG HIT {hit_count}] path={path}.{key}")
                print(f"    key={key}")
                print(f"    value={summarize_value(value)}")

            walk(value, f"{path}.{key}")

    for page_idx, res in enumerate(visual_predict_res, start=1):
        print(f"\n[SEAL DEBUG] page={page_idx}")
        print(f"[SEAL DEBUG] raw res type={type(res).__name__}")

        res_json = get_json_like(res)

        if isinstance(res_json, dict):
            print(f"[SEAL DEBUG] top keys={list(res_json.keys())}")
            walk(res_json, f"page_{page_idx}")
        else:
            print("[SEAL DEBUG] cannot convert page result to dict/json")

    print("========== SEAL DEBUG END ==========\n")


def extract_seal_info_from_visual_res(visual_predict_res):
    seal_detected = False
    seal_texts = []
    seal_scores = []
    seal_source = ""

    def add_text(text):
        if text is None:
            return
        text = str(text).strip()
        if text:
            seal_texts.append(text)

    def add_texts(texts):
        if not texts:
            return
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            add_text(text)

    def add_scores(scores):
        if not scores:
            return
        if isinstance(scores, (int, float)):
            scores = [scores]
        for score in scores:
            try:
                seal_scores.append(float(score))
            except Exception:
                pass

    def has_non_empty_box(value):
        if value is None:
            return False
        if isinstance(value, list):
            return len(value) > 0
        try:
            return len(value) > 0
        except Exception:
            return str(value).strip() not in {"", "[]", "None"}

    def text_has_school_or_seal_signal(text):
        t = str(text or "").lower()

        keywords = [
            "stamp",
            "seal",
            "official",
            "certify",
            "principal",
            "school",
            "academy",
            "university",
            "bpk",
            "smak",
            "penabur",
            "gading serpong",
            "公章",
            "印章",
            "盖章",
        ]

        return any(k in t for k in keywords)

    def text_has_signature_context(text):
        t = str(text or "").lower()

        context_words = [
            "officially certify",
            "certify",
            "official transcript",
            "academic record",
            "principal",
            "headmaster",
            "signed",
            "signature",
        ]

        return any(k in t for k in context_words)

    for page_idx, res in enumerate(visual_predict_res, start=1):
        if not isinstance(res, dict):
            continue

        layout = res.get("layout_parsing_result")
        layout_json = get_result_json(layout)

        if not isinstance(layout_json, dict):
            continue

        # 1. 优先读取 seal_res_list
        seal_res_list = layout_json.get("seal_res_list", [])

        if isinstance(seal_res_list, list) and seal_res_list:
            seal_detected = True
            seal_source = "seal_res_list"

            for seal_res in seal_res_list:
                seal_res_json = get_result_json(seal_res) or seal_res
                if not isinstance(seal_res_json, dict):
                    continue

                add_texts(seal_res_json.get("rec_texts"))
                add_scores(seal_res_json.get("rec_scores"))

                for box_key in ["dt_polys", "rec_polys", "rec_boxes"]:
                    if has_non_empty_box(seal_res_json.get(box_key)):
                        seal_detected = True

        # 2. fallback: 从 parsing_res_list 里找疑似印章 image block
        parsing_res_list = layout_json.get("parsing_res_list", [])

        if isinstance(parsing_res_list, list):
            for idx, block in enumerate(parsing_res_list):
                if not isinstance(block, dict):
                    continue

                block_label = str(block.get("block_label", "")).lower()
                block_content = str(block.get("block_content", "")).strip()

                if block_label != "image":
                    continue

                if not block_content:
                    continue

                # 防止把页眉 logo 当成印章：
                # 只看页面后半部分的 image block
                if idx < len(parsing_res_list) * 0.5:
                    continue

                prev_text = "\n".join(
                    str(x.get("block_content", ""))
                    for x in parsing_res_list[max(0, idx - 5):idx]
                    if isinstance(x, dict)
                )

                next_text = "\n".join(
                    str(x.get("block_content", ""))
                    for x in parsing_res_list[idx + 1:idx + 5]
                    if isinstance(x, dict)
                )

                surrounding_text = prev_text + "\n" + block_content + "\n" + next_text

                if (
                    text_has_school_or_seal_signal(block_content)
                    and text_has_signature_context(surrounding_text)
                ):
                    seal_detected = True

                    if not seal_source:
                        seal_source = "parsing_res_list_image_fallback"

                    add_text(block_content)

    # 去重
    unique_texts = []
    seen = set()
    for text in seal_texts:
        text = str(text).strip()
        if text and text not in seen:
            seen.add(text)
            unique_texts.append(text)

    return {
        "seal_detected": seal_detected,
        "seal_texts": unique_texts,
        "seal_scores": seal_scores,
        "max_seal_score": max(seal_scores) if seal_scores else None,
        "seal_source": seal_source,
    }



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
            "content": (
                "You are an information extraction engine. "
                "Return ONLY one valid JSON object. "
                "Do not output markdown. "
                "Do not output explanations. "
                "Do not output <think> tags."
            ),
        },
        {
            "role": "user",
            "content": prompt + "\n\n/no_think",
        },
    ],
    temperature=0,
    max_tokens=1536,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },
)
    content = response.choices[0].message.content.strip()
    return content

def parse_llm_json(content, field_meta):
    raw = (content or "").strip()

    # 去掉 Qwen3 可能输出的 <think>...</think>
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 去掉 markdown 代码块
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]

    if raw.endswith("```"):
        raw = raw[:-3]

    raw = raw.strip()

    # 如果 JSON 前后还有多余文本，只截取第一个 { 到最后一个 }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    field_names = [
        item.get("name", "")
        for item in field_meta
        if item.get("name", "").strip()
    ]
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
        print(f"[清洗后的输出] {raw}")

    return skeleton




# main method
def extract_fields(input_path, template, field_meta):

    t0 = time.time()

    # 1. OCR
    visual_predict_res = pipeline.visual_predict(
        input=input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
    )

    debug_print_seal_structure(visual_predict_res)
    print(f"[耗时] OCR: {time.time() - t0:.2f}s")

    t1 = time.time()

    # 2. OCR 结果转文本
    document_text = visual_result_to_text(visual_predict_res)

    seal_info = extract_seal_info_from_visual_res(visual_predict_res)

    print(f"[印章检测] seal_detected: {seal_info['seal_detected']}")
    print(f"[印章检测] seal_texts: {seal_info['seal_texts']}")
    print(f"[印章检测] max_seal_score: {seal_info['max_seal_score']}")

    visual_signals = [
        "[OCR Visual Signals]",
        f"school stamp detected: {'Yes' if seal_info['seal_detected'] else 'Unknown'}",
    ]

    if seal_info["seal_texts"]:
        visual_signals.append(
            "seal recognized text: " + " | ".join(seal_info["seal_texts"])
        )

    if seal_info["max_seal_score"] is not None:
        visual_signals.append(
            f"seal max recognition score: {seal_info['max_seal_score']:.4f}"
        )

    document_text = "\n".join(visual_signals) + "\n\n" + document_text

    print(f"[耗时] OCR结果转文本: {time.time() - t1:.2f}s")
    print(f"[长度] document_text chars: {len(document_text)}")

    t2 = time.time()

    # 3. 根据模板生成 instruction
    instruction = build_text_prompt_from_template(template, field_meta)

    # 4. document_text + instruction 直接喂给 LLM
    llm_raw_output = call_llm_with_text(document_text, instruction)

    print(f"[耗时] LLM: {time.time() - t2:.2f}s")

    print(f"[耗时] 总计: {time.time() - t0:.2f}s")

    return llm_raw_output, seal_info


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

def safe_filename(value):
    value = str(value or "").strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("_")
    return value or "batch_result"

def clean_application_id(value):
    text = str(value or "").strip()

    match = re.search(r"\bUG\d{6,}\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()

    text = re.sub(
        r"(?i)^application\s*(id|number)\s*[:：]\s*",
        "",
        text
    ).strip()

    return text


def clean_person_name(value):
    text = str(value or "").strip()

    text = re.sub(
        r"(?i)^(name|applicant name|student name|full name)\s*[:：]\s*",
        "",
        text
    ).strip()

    text = re.sub(r"\s+", " ", text)

    return text


def get_value_from_result(result, aliases):
    chat_data = result.get("chat_data", {}) or {}

    lower_map = {
        str(k).strip().lower(): k
        for k in chat_data.keys()
    }

    for alias in aliases:
        real_key = lower_map.get(str(alias).strip().lower())
        if real_key is None:
            continue

        value = str(chat_data.get(real_key, "")).strip()
        if value:
            return value

    return ""


def get_value_from_results(results_by_type, aliases, preferred_doc_types=None):
    if preferred_doc_types is None:
        preferred_doc_types = [
            "application_form",
            "passport",
            "transcript",
            "diploma_certificate",
            "english_language",
        ]

    for doc_type in preferred_doc_types:
        result = results_by_type.get(doc_type)
        if not result:
            continue

        value = get_value_from_result(result, aliases)
        if value:
            return value

    return ""


def build_output_base_name(results_by_type, fallback_name=None):
    application_id = get_value_from_results(
        results_by_type,
        [
            "application id",
            "application number",
            "application no",
            "applicant id",
            "student id",
        ],
        preferred_doc_types=["application_form"]
    )

    application_id = clean_application_id(application_id)

    applicant_name = get_value_from_results(
        results_by_type,
        [
            "name",
            "applicant name",
            "student name",
            "full name",
            "passport name",
        ],
        preferred_doc_types=[
            "application_form",
            "passport",
            "transcript",
            "diploma_certificate",
            "english_language",
        ]
    )

    applicant_name = clean_person_name(applicant_name)

    parts = []

    if application_id:
        parts.append(application_id)

    if applicant_name:
        parts.append(applicant_name)

    if parts:
        return safe_filename("_".join(parts))

    return safe_filename(fallback_name)

def process_folder(input_dir, output_root=None, output_base_name=None):
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

    base_name = build_output_base_name(
        results_by_type,
        fallback_name=output_base_name
    )

    logs.append(f"[最终输出前缀] {base_name}")

    combined_output_path = os.path.join(
        output_root,
        f"{base_name}_combined_result.txt"
    )

    save_combined_result_to_txt(results_by_type, combined_output_path)

    report_output_path = os.path.join(
        output_root,
        f"{base_name}_report.html"
    )

    save_validation_report_to_html(
        results_by_type=results_by_type,
        output_file=report_output_path,
        doc_type_order=DOC_TYPE_ORDER,
    )

    logs.append(f"[报告] HTML 校验报告已生成: {report_output_path}")

    return combined_output_path, report_output_path, results_by_type, logs



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

    llm_raw_output, seal_info = extract_fields(
    input_path=input_file,
    template=template,
    field_meta=field_meta
    )

    print(f"[原始结果类型] {type(llm_raw_output).__name__}")
    print("\n================ LLM RAW OUTPUT ================\n")
    print(llm_raw_output)
    print("\n================ END OF OUTPUT ================\n")

    chat_data = parse_llm_json(llm_raw_output, field_meta)
    doc_type = template.get("doc_type", "unknown")

    if doc_type == "transcript":
        if seal_info.get("seal_detected"):
            chat_data["school stamp present"] = "Yes"
        else:
            chat_data["school stamp present"] = "Unknown"

        seal_text = ""
        if seal_info.get("seal_texts"):
            seal_text = " | ".join(seal_info["seal_texts"])
            chat_data["seal recognized text"] = seal_text

        if seal_info.get("max_seal_score") is not None:
            chat_data["seal max recognition score"] = f"{seal_info['max_seal_score']:.4f}"

        if seal_info.get("seal_source"):
            chat_data["seal source"] = seal_info.get("seal_source")

        school_name = (
            chat_data.get("institution name")
            or chat_data.get("school name")
            or chat_data.get("institution")
            or chat_data.get("name of institution")
            or chat_data.get("issuing school")
            or ""
        )

        if not seal_info.get("seal_detected"):
            chat_data["seal school name match"] = "Unknown"
            chat_data["seal school name match reason"] = "未检测到印章，无法确认印章是否属于成绩单学校。"

        elif not seal_text:
            chat_data["seal school name match"] = "Unknown"
            chat_data["seal school name match reason"] = "检测到印章区域，但未识别出可用于比较的印章文字。"

        elif not school_name:
            chat_data["seal school name match"] = "Unknown"
            chat_data["seal school name match reason"] = "未抽取到成绩单学校名，无法和印章文字比较。"

        else:
            check_prompt = f"""
    You are checking whether the recognized school stamp/seal text belongs to the same institution as the transcript.

    Transcript institution name:
    {school_name}

    Recognized seal/stamp text:
    {seal_text}

    Decide whether they refer to the same school or institution.

    Rules:
    - Return "Yes" if they are clearly consistent.
    - Return "No" if they clearly refer to different schools or institutions.
    - Return "Unknown" if the seal text is too incomplete, too ambiguous, or OCR quality is insufficient.
    - Be tolerant of OCR errors, missing spaces, abbreviations, and partial words.
    - Do not over-penalize abbreviations such as BPK, SMAK, PENABUR, campus/location names like Gading Serpong.

    Return JSON only:
    {{
    "match": "Yes",
    "reason": "short reason"
    }}
    """

            try:
                seal_check_response = client.chat.completions.create(
                    model=LLM_CONFIG["model_name"],
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a document consistency checker. "
                                "Return ONLY one valid JSON object. "
                                "Do not output markdown. "
                                "Do not output explanations outside JSON. "
                                "Do not output <think> tags."
                            ),
                        },
                        {
                            "role": "user",
                            "content": check_prompt + "\n\n/no_think",
                        },
                    ],
                    temperature=0,
                    max_tokens=512,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        }
                    },
                )

                seal_check_raw = seal_check_response.choices[0].message.content.strip()
                seal_check_raw = re.sub(
                    r"<think>.*?</think>",
                    "",
                    seal_check_raw,
                    flags=re.DOTALL
                ).strip()

                start = seal_check_raw.find("{")
                end = seal_check_raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    seal_check_raw = seal_check_raw[start:end + 1]

                seal_check_data = json.loads(seal_check_raw)

                seal_match = str(seal_check_data.get("match", "Unknown")).strip()
                seal_reason = str(seal_check_data.get("reason", "")).strip()

                if seal_match not in {"Yes", "No", "Unknown"}:
                    seal_match = "Unknown"

                chat_data["seal school name match"] = seal_match
                chat_data["seal school name match reason"] = seal_reason

            except Exception as e:
                chat_data["seal school name match"] = "Unknown"
                chat_data["seal school name match reason"] = f"LLM 判断印章学校名一致性失败: {e}"

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
        "seal_info": seal_info,
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
