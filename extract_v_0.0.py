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


# 主函数
def extract_fields(input_path, field_list):

    pipeline = PPChatOCRv4Doc()

    # 1️⃣ OCR
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

    # 2️⃣ LLM 抽取
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
    input_file = os.getenv("INPUT_FILE", os.path.join(os.path.dirname(__file__), "data", "sample.png"))

    fields = [
        "驾驶室准乘人数",
        "车辆型号",
        "发动机号码",
    ]

    result = extract_fields(input_file, fields)

    print("RESULT:", result)
    
    chat_data = result.get("chat_res", {})

 
    print("CHAT_DATA:", result.get("chat_res"))

 
    # 写入 txt
    output_file = os.path.splitext(input_file)[0] + ".txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for key in fields:
            value = chat_data.get(key, "未找到")
            f.write(f"{key}: {value}\n")

    print(f"\n结果已保存到: {output_file}")
