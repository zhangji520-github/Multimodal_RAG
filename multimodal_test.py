import dashscope
from http import HTTPStatus
import json

# 文本向量测试
# input_texts = "机器学习的相关论文"
# resp_txt = dashscope.TextEmbedding.call(
#     model="text-embedding-v4",
#     input=input_texts,
#     text_type="query",
#     instruct="Given a research paper query, retrieve relevant research paper"     # 使用任务指令提升效果 (instruct)
# )

# if resp_txt.status_code == HTTPStatus.OK:
#     # resp.output 是字典类型，需要用字典方式访问
#     embedding = resp_txt.output['embeddings'][0]['embedding']
#     print(f'文本向量测试成功')
#     print(f'维度为 {len(embedding)}')
#     print(f'前5个值: {embedding[:5]}')
# else:
#     print(f"请求失败: {resp_txt.code} - {resp_txt.message}")


# 图片向量测试 多模态
print("图片向量测试 多模态")
image = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"
input = [{"image": image}]
resp_img = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=input,
)

# 检查响应状态
if resp_img.status_code == HTTPStatus.OK:
    print("图片✅ 请求成功！")
    # 提取 embedding 数据
    embedding = resp_img.output['embeddings'][0]['embedding']
    usage = resp_img.usage

    print(f'维度为 {len(embedding)}')
    print(f'前5个值: {json.dumps(embedding[:5], indent=4)}')          # json打印更美观
    print(f'使用tokens: {usage["total_tokens"]}')
    print(f'类型为 {resp_img.output["embeddings"][0]["type"]}')   # image or text or video
# 视频向量测试 多模态
print("视频向量测试 多模态")
video = r"F:\workspace\langgraph_project\Adaptive_RAG\datas\new+video.mp4"
input = [{"video": video}]
resp_video = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",
    input=input,
)
# 检查响应状态
if resp_video.status_code == HTTPStatus.OK:
    print("视频✅ 请求成功！")

    # 提取 embedding 数据
    embedding = resp_video.output['embeddings'][0]['embedding']
    usage = resp_video.usage

    print(f'维度为 {len(embedding)}')
    print(f'前5个值: {json.dumps(embedding[:5], indent=4)}')          # json打印更美观
    print(f'image_tokens: {usage['input_tokens_details']['image_tokens']},text_tokens: {usage['input_tokens_details']['text_tokens']},total_tokens: {usage["total_tokens"]}')
    print(f'类型为 {resp_video.output["embeddings"][0]["type"]}')   # image or text or video