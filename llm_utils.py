from langchain_openai import ChatOpenAI
from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL, LOCAL_BASE_URL, DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, \
    MODELSCOPE_API_KEY, MODELSCOPE_BASE_URL, QWEN_BASE_URL, GLM_API_KEY, GLM_BASE_URL, SILICONFLOW_API_KEY,SILICONFLOW_BASE_URL
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

# OpenAI SDK 初始化

# OpenAI 模型
llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# OpenAI Embedding 模型
openai_embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
    dimensions=1024  # 将维度设置为1024以匹配Milvus集合schema
    )

qwen_embeddings = DashScopeEmbeddings(
    model="text-embedding-v4", 
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
    # DashScopeEmbeddings 内部会自动处理批处理大小限制
)

# 全模态大模型
# multiModal_llm = ChatOpenAI(
#     model='qwen-omni-turbo',
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=QWEN_BASE_URL,
#     streaming=True
# )

# 全模态私有化部署大模型
# multiModal_llm = ChatOpenAI(
#     model='qwen-omni-7b',
#     api_key=os.getenv("xxx"),
#     base_url=LOCAL_BASE_URL,
# )

# qwen 家族模型
# llm = ChatOpenAI(
#     model='qwen3-235b-a22b',
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=QWEN_BASE_URL,
#     streaming=True,
#     extra_body={
#         "enable_search": True,  # 启用搜索功能
#         "search_options": {
#             "forced_search": False
#         },
#         "enable_thinking": False  # 启用思考功能
#     }
# )

# llm = ChatOpenAI(
#     model='qwen3-32b',
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=QWEN_BASE_URL,
#     streaming=True,
#     extra_body={
#         "enable_search": True,  # 启用搜索功能
#         "search_options": {
#             "forced_search": False
#         },
#         "enable_thinking": False  # 启用思考功能
#     }
# )

# deepseek
# llm = ChatOpenAI(
#     model='deepseek-chat',
#     # model='deepseek-reasoner'
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
# )

# glm-asr 模型
# llm = ChatOpenAI(
#     # model='glm-asr',
#     model='glm-4-air-250414',
#     api_key=GLM_API_KEY,
#     base_url=GLM_BASE_URL,
# )


# # GLM 全模态
# multiModal_llm = ChatOpenAI(
#     model='glm-4v-plus-0111',
#     api_key=GLM_API_KEY,
#     base_url=GLM_BASE_URL,
# )

