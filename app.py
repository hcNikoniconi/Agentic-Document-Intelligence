import os
import shutil
import uuid
import gradio as gr

import importlib
import extract_v_0_5
import validator

import re

BASE_DIR = os.path.abspath(os.getenv("APP_BASE_DIR", os.path.dirname(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", os.path.join(BASE_DIR, "output", "test"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)



def get_extract_module():
    """
    每次点击按钮时重新加载 validator.py 和 extract_v_0_5.py。
    这样修改后端逻辑后，不需要重启 Gradio。
    """
    importlib.reload(validator)
    return importlib.reload(extract_v_0_5)
    
def run_extraction(uploaded_file):
    if uploaded_file is None:
        return "请先上传 PDF 文件。", "", None

    try:
        src_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name

        original_name = os.path.basename(src_path)
        unique_name = f"{uuid.uuid4().hex}_{original_name}"
        saved_input_path = os.path.join(UPLOAD_DIR, unique_name)

        shutil.copy(src_path, saved_input_path)

        extract = get_extract_module()

        raw_output_file, result = extract.process_one_file(
            saved_input_path,
            output_root=OUTPUT_ROOT
        )

        
        safe_output_path = raw_output_file

        preview_text = result["preview_text"]
        status_text = (
            f"处理完成\n"
            f"文档类型: {result['doc_type']}\n"
            f"模板: {os.path.basename(result['template_file'])}\n"
            f"下载文件: {safe_output_path}"
        )

        return status_text, preview_text, safe_output_path

    except Exception as e:
        return f"处理失败: {e}", "", None

def safe_filename(value):
    value = str(value or "").strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("_")
    return value


def guess_uploaded_folder_name(uploaded_files):
    """
    尝试从 gr.File(file_count="directory") 返回的路径里推断原始文件夹名。

    情况 1：
    /tmp/gradio/.../APP000001_Sample Applicant/passport.pdf
    -> APP000001_Sample Applicant

    情况 2：
    只返回 /tmp/gradio/.../passport.pdf
    -> 无法知道原始文件夹名，只能 fallback
    """
    if not uploaded_files:
        return ""

    paths = []
    for f in uploaded_files:
        path = f if isinstance(f, str) else getattr(f, "name", "")
        if path:
            paths.append(path)

    print("[目录上传 DEBUG] uploaded paths:")
    for p in paths[:20]:
        print("  ", p)

    if not paths:
        return ""

    # 先找公共父目录
    common_dir = os.path.commonpath(paths)

    # 如果 common_dir 本身不是文件夹，就取父目录
    if os.path.isfile(common_dir):
        common_dir = os.path.dirname(common_dir)

    folder_name = os.path.basename(os.path.normpath(common_dir))

    # 避免拿到 gradio/temp/tmp 这种无意义名字
    bad_names = {
        "",
        "tmp",
        "temp",
        "gradio",
        "upload",
        "uploads",
        "files",
    }

    if folder_name.lower() in bad_names:
        return ""

    # 如果是 Gradio 随机缓存目录，也不要用
    if re.fullmatch(r"[a-f0-9]{16,}", folder_name.lower()):
        return ""

    return folder_name

def run_batch_extraction(uploaded_files):
    if not uploaded_files:
        return "请先上传一个学生文件夹。", "", "", None, None

    try:
        guessed_folder_name = guess_uploaded_folder_name(uploaded_files)
        output_base_name = safe_filename(guessed_folder_name)

        if not output_base_name:
            output_base_name = f"batch_{uuid.uuid4().hex[:8]}"

        batch_dir = os.path.join(
            UPLOAD_DIR,
            f"tmp_{output_base_name}_{uuid.uuid4().hex[:8]}"
        )
        os.makedirs(batch_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            src_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name

            if not src_path.lower().endswith(".pdf"):
                continue

            original_name = os.path.basename(src_path)
            unique_name = f"{uuid.uuid4().hex}_{original_name}"
            saved_input_path = os.path.join(batch_dir, unique_name)

            shutil.copy(src_path, saved_input_path)

        extract = get_extract_module()

        combined_output_file, report_file, results_by_type, logs = extract.process_folder(
            batch_dir,
            output_root=OUTPUT_ROOT,
            output_base_name=output_base_name
        )

        ordered_doc_types = [
            "passport",
            "application_form",
            "transcript",
            "diploma_certificate",
            "english_language",
        ]

        preview_parts = []
        for doc_type in ordered_doc_types:
            if doc_type in results_by_type:
                preview_parts.append(results_by_type[doc_type]["preview_text"])

        preview_text = "\n\n".join(preview_parts) if preview_parts else "未识别到可用文件。"

        status_text = (
        f"批量处理完成\n"
        f"[自动识别文件夹名] {guessed_folder_name or '[未识别到，改用 application id + name 命名]'}\n"
        + "\n".join(logs)
    )

        report_preview = ""
        if report_file and os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_preview = f.read()
        else:
            report_preview = "未生成校验报告。"

        return status_text, preview_text, report_preview, combined_output_file, report_file

    except Exception as e:
        return f"处理失败: {e}", "", "", None, None


with gr.Blocks(title="Agentic Document Intelligence") as demo:
    gr.Markdown("## Agentic Document Intelligence")

    with gr.Tab("单文件处理"):
        gr.Markdown("上传一个 PDF，自动匹配模板并输出结果 txt。")

        file_input = gr.File(label="上传 PDF", file_types=[".pdf"], type="filepath")
        submit_btn = gr.Button("开始抽取")

        status_box = gr.Textbox(label="状态", lines=6, interactive=False)
        preview_box = gr.Textbox(label="结果预览", lines=20, interactive=False)
        download_file = gr.File(label="下载结果 txt")

        submit_btn.click(
            fn=run_extraction,
            inputs=[file_input],
            outputs=[status_box, preview_box, download_file]
        )

    with gr.Tab("批量处理"):
        gr.Markdown(
        "上传多个 PDF，系统会自动识别 passport / application form / transcript / diploma / english language，"
        "并汇总输出一个 txt，同时生成一份 HTML 校验报告。"
    )

        batch_input = gr.File(
            label="上传学生文件夹",
            file_count="directory",
            type="filepath"
        )

        batch_btn = gr.Button("开始批量抽取")

        batch_status_box = gr.Textbox(
            label="状态",
            lines=10,
            interactive=False
        )

        batch_preview_box = gr.Textbox(
            label="汇总预览",
            lines=25,
            interactive=False
        )

        gr.Markdown("### 校验报告预览")

        batch_report_box = gr.HTML(
            value=""
        )

        batch_download_file = gr.File(
            label="下载汇总 txt"
        )

        batch_download_report = gr.File(
            label="下载校验报告 html"
        )

        batch_btn.click(
            fn=run_batch_extraction,
            inputs=[batch_input],
            outputs=[
                batch_status_box,
                batch_preview_box,
                batch_report_box,
                batch_download_file,
                batch_download_report,
            ]
        )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("APP_HOST", "127.0.0.1"),
        server_port=int(os.getenv("APP_PORT", "7860")),
        allowed_paths=[
            os.path.abspath(OUTPUT_ROOT),
            os.path.abspath(UPLOAD_DIR),   # 可留可不留，留着排查更方便
        ],)
