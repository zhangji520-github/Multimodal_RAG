import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 修正拼写错误
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")  # 修正获取的环境变量
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY")
MODELSCOPE_BASE_URL = os.getenv("MODELSCOPE_BASE_URL")

GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = os.getenv("GLM_BASE_URL")

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")


SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")


## Milvus 配置
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "multimodal_rag"