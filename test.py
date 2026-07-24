import os
from paddleocr import PPChatOCRv4Doc

# 解决线程问题（必须）
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# LLM config（千帆 API 示例）
chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": os.getenv("MODEL_NAME", "ernie-3.5-8k"),
    "base_url": os.getenv("MODEL_BASE_URL", "https://qianfan.baidubce.com/v2"),
    "api_type": "openai",
    "api_key": os.getenv("MODEL_API_KEY", ""),
}


# embedding（暂时不用）
# retriever_config = {
#     "module_name": "retriever",
#     "model_name": "embedding-v1",
#     "base_url": "https://qianfan.baidubce.com/v2",
#     "api_type": "qianfan",
#     "api_key": "你的api_key",
# }


# 本地多模态模型（暂时不用）
# mllm_chat_bot_config = {
#     "module_name": "chat_bot",
#     "model_name": "PP-DocBee2",
#     "base_url": "http://127.0.0.1:8080/",
#     "api_type": "openai",
#     "api_key": "api_key",
# }



# 初始化
pipeline = PPChatOCRv4Doc()

# step1：（OCR）
visual_predict_res = pipeline.visual_predict(
    input=os.getenv("INPUT_FILE", os.path.join(os.path.dirname(__file__), "data", "sample.png")),
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])


# 向量构建（暂时关闭）
# vector_info = pipeline.build_vector(
#     visual_info_list,
#     flag_save_bytes_vector=True,
#     retriever_config=retriever_config
# )


# 多模态预测（暂时关闭）
# mllm_predict_res = pipeline.mllm_pred(
#     input=os.getenv("INPUT_FILE"),
#     key_list=["驾驶室准乘人数"],
#     mllm_chat_bot_config=mllm_chat_bot_config,
# )
# mllm_predict_info = mllm_predict_res["mllm_res"]


# step2：直接 LLM 信息抽取

chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=None,              # 关闭检索
    mllm_predict_info=None,        # 关闭多模态
    chat_bot_config=chat_bot_config,
    retriever_config=None,
)

print("\n====== 抽取结果 ======")
print(chat_result)
