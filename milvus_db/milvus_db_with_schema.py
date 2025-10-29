"""Milvus向量数据库操作模块.
提供Milvus向量数据库的集合创建、连接和文档添加功能。
支持稠密向量和稀疏向量的混合索引。
"""
import os
import sys
from typing import List, Optional, Dict

# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from splitters.splitter_md import MarkdownDirSplitter
from langchain_core.documents import Document
from utils.embeddings_utils import process_item_with_guard, limiter, RETRY_ON_429, MAX_429_RETRIES, BASE_BACKOFF
from langchain_milvus import Milvus
from pymilvus import DataType, Function, FunctionType, MilvusClient
from env_utils import COLLECTION_NAME, MILVUS_URI
from utils.embeddings_utils import image_to_base64
from utils.common_utils import get_surrounding_text_content
from langchain_core.messages import HumanMessage  
import logging
import time
import random
from llm_utils import qwen3_max

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MilvusVectorSave:
    """
    建立一个 Milvus 集合 在schema中保留了未来需要的字段 使用混合索引，包含稀疏向量和稠密向量字段 
    """
    def __init__(self):
        # 类型注解：明确声明属性类型，提供IDE智能提示和类型检查
        self.vector_stored_saved: Optional[Milvus] = None
        self.client = MilvusClient(uri=MILVUS_URI, user='root', password='Milvus')

    def create_collection(self,collection_name: str = COLLECTION_NAME, uri: str = MILVUS_URI, is_first: bool = False):
        """创建一个collection milvus + langchain"""

        # 2. 定义schema
        schema = self.client.create_schema()

        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True, description="主键")
        schema.add_field("category", DataType.VARCHAR, max_length=1000, description="对应元数据的'embedding_type'")     
        schema.add_field("filename", DataType.VARCHAR, max_length=1000, description="对应元数据的'source',文件名,带后缀")     
        schema.add_field("filetype", DataType.VARCHAR, max_length=1000, description="对应元数据的'filetype',pdf或者md")     

        schema.add_field("title", DataType.VARCHAR, max_length=1000, enable_analyzer=True, 
                        analyzer_params={'tokenizer': 'jieba', 'filter': ['cnalphanumonly']}, description="对应元数据的Header")
        schema.add_field("text", DataType.VARCHAR, max_length=10000, enable_analyzer=True,
                        analyzer_params={'tokenizer': 'jieba', 'filter': ['cnalphanumonly']}, description="对应每个文本块的内容") 
        schema.add_field("image_path", DataType.VARCHAR, max_length=2000, description="图片文件的本地路径，仅图片类型数据使用")

        schema.add_field("title_sparse", DataType.SPARSE_FLOAT_VECTOR, description="标题的稀疏向量嵌入")
        schema.add_field("text_content_sparse", DataType.SPARSE_FLOAT_VECTOR, description="文档块的稀疏向量嵌入")
        schema.add_field("text_content_dense", DataType.FLOAT_VECTOR, dim=1024, description="文档块的稠密向量嵌入")

        logger.info(f'🐶添加schema完成,共添加{len(schema.fields)}个字段')

        # 3 稀疏向量需要的bm25函数
        title_bm25_function = Function(
            name = "title_bm25_emb",
            input_field_names=["title"], # 需要进行文本到稀疏向量转换的 VARCHAR 字段名称。
            output_field_names=["title_sparse"], # 存储内部生成稀疏向量的字段名称。
            function_type=FunctionType.BM25 # 要使用的函数类型。
        )
        schema.add_function(title_bm25_function)          # bm25 此功能会根据文本的语言标识自动应用相应的分析器

        content_bm25_function = Function(
            name = "text_content_bm25_emb",
            input_field_names=["text"], # 需要进行文本到稀疏向量转换的 VARCHAR 字段名称。
            output_field_names=["text_content_sparse"], # 存储内部生成稀疏向量的字段名称。
            function_type=FunctionType.BM25 # 要使用的函数类型。
        )
        schema.add_function(content_bm25_function)          # bm25 此功能会根据文本的语言标识自动应用相应的分析器

        # 4 创建索引参数对象
        try:
            logger.info("开始创建索引参数...")
            index_params = self.client.prepare_index_params()

            # 主键索引
            index_params.add_index(
                field_name="id",
                index_type="AUTOINDEX",
            )

            # 稀疏向量索引 - 标题
            index_params.add_index(
                field_name="title_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",  # 算法选择
                    "bm25_k1": 1.2,  # 词频饱和度控制参数
                    "bm25_b": 0.75  # 文档长度归一化参数
                }
            )

            # 稀疏向量索引 -文本块
            index_params.add_index(
                field_name="text_content_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",  # 算法选择
                    "bm25_k1": 1.2,  # 词频饱和度控制参数
                    "bm25_b": 0.75  # 文档长度归一化参数
                }
            )

            # 稠密向量索引 - 文本块
            index_params.add_index(
                field_name="text_content_dense",
                index_type="HNSW",  # 适合稠密向量的索引类型
                metric_type="COSINE",  # 余弦相似度
                params={
                    "M": 16,  # HNSW图中每个节点的最大连接数
                    "efConstruction": 200  # 构建索引时的搜索候选数
                }
            )

            logger.info("🐶成功添加稀疏向量索引和稠密向量索引")

        except Exception as e:
            logger.error(f"🐶创建索引参数失败: {e}")

        #  5. 创建集合
        # 检查集合是否已存在，如果存在先释放collection，然后再删除索引和集合  
        if is_first:  
            if COLLECTION_NAME in self.client.list_collections():
                self.client.release_collection(collection_name=COLLECTION_NAME)
                # self.client.drop_index(collection_name=COLLECTION_NAME, index_name="sparse_inverted_index")
                # self.client.drop_index(collection_name=COLLECTION_NAME, index_name="dense_vector_index")
                self.client.drop_collection(collection_name=COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"🐶成功创建集合: {COLLECTION_NAME}")
    
    @staticmethod
    def doc_to_dict(docs: List[Document]) -> List[Dict]:
        """
        将Document列表转换为指定格式的字典
        Args:
            docs: 包含 Document 对象的列表  Document(page_content='', metadata={'source': 'pdf', 'embedding_type': 'text/image/video'}, 'image_path': 'path/to/image.jpg', 'Header1':'...', 'Header2':'...', 'Header3':'...')
        Returns:
            List[Dict]: 指定格式的字典列表
        """
        result_dict = []

        for doc in docs:
            # 初始化一个空字典存储当前文档信息
            doc_dict = {}
            metadata = doc.metadata
            
            # 1. 提取text (仅当embedding_type为text)
            if metadata.get('embedding_type') == 'text':
                doc_dict['text'] = doc.page_content
            else:
                doc_dict['text'] = ''       # # 图片类型初始设置为空字符串
            
            # 2. 提取 category (embedding_type)
            doc_dict['category'] = metadata.get('embedding_type', '')
            
            # 3. 提取 filename 和 filetype (pdf/md 也就是 source的文件名后缀)
            source = metadata.get('source', '')
            doc_dict['filename'] = source
            _, ext = os.path.splitext(source)
            doc_dict['filetype'] = ext[1:].lower() if ext.startswith('.') else ext.lower()
            
            # 4.  图片专用处理 提取 image_path (仅当 embedding_type 为 'image' 时)
            if metadata.get('embedding_type') == 'image':
                doc_dict['image_path'] = doc.page_content        # 图片路径存储在image_path
                doc_dict['text'] = '图片'  # 覆盖之前的空字符串，设置为"图片"
            else:
                doc_dict['image_path'] = ''
            
            # 5. 对于文本块  提取 title(拼接所有的 Header) 与内容(doc.page_content) 存储到 text 字段
            headers = []
            # 假设 Header 的键可能为 'Header 1', 'Header 2', 'Header 3' 等，我们按层级顺序拼接
            # 我们需要先收集所有存在的 Header 键，并按层级排序
            header_keys = [key for key in metadata.keys() if key.startswith('Header')]              # ['Header 1', 'Header 3']
            # 按 Header 后的数字排序，确保层级顺序
            header_keys_sorted = sorted(header_keys, key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else x)

            for key in header_keys_sorted:
                value = metadata.get(key, '').strip()
                if value:  # 只添加非空的 Header 值
                    headers.append(value)
            
            # 将所有非空的 Header 值用连字符或空格连接起来
            doc_dict['title'] = ' --> '.join(headers) if headers else ''  # 你也可以用其他连接符，如空格
            # 对文本块处理：拼接标题和内容
            if metadata.get('embedding_type') == 'text':
                if doc_dict['title']:
                    doc_dict['text'] = doc_dict['title'] + ':' + doc.page_content
                else:
                    doc_dict['text'] = doc.page_content
            
            # 6. 将doc_dict添加到result_dict中
            result_dict.append(doc_dict)
            
        return result_dict
    
    def write_to_milvus(self, processed_data: List[Dict]):
        """
        把数据写入到Milvus中
        :param processed_data:
        :return:
        """
        if not processed_data:
            logger.warning("🐶没有需要写入的数据")
            return
        
        # 数据清洗：确保text字段不超过最大长度
        MAX_TEXT_LENGTH = 10000
        for item in processed_data:
            text = item.get('text', '')
            if len(text) > MAX_TEXT_LENGTH:
                logger.warning(f"⚠️ 文本超长({len(text)}字符)，已截断至{MAX_TEXT_LENGTH}字符: {text[:50]}...")
                item['text'] = text[:MAX_TEXT_LENGTH]
        
        try:
            insert_res = self.client.insert(collection_name=COLLECTION_NAME, data=processed_data)
            print(f"[Milvus] 成功写入 {len(processed_data)} 条数据.IDs 示例: {insert_res['ids'][:5]}")
        except Exception as e:
            logger.error(f"🐶写入Milvus失败: {e}")
            raise e

    @staticmethod
    def generate_image_description(data_list:List[Dict]):
        """
        为文档中包含图片的条目(image_path 字段非空)生成一段基于上下文的、简洁的多模态文本描述 以便后续可以将这段描述用于向量化(embedding)并存入向量数据库 Milvus。

        参数:
            data_list: 包含字典的列表

        返回:
            包含完整结果的新列表
        """
        for index, item in enumerate(data_list):
            if item.get('image_path'):  # 检查是否为图片字典
                # 获取前后文本内容
                prev_text, next_text = get_surrounding_text_content(data_list, index)
                
                # 打印调试信息
                logger.info(f"\n{'='*50}")
                logger.info(f"正在处理图片: {item.get('image_path')}")
                logger.info(f"前文内容: {prev_text[:100] if prev_text else 'None'}...")
                logger.info(f"后文内容: {next_text[:100] if next_text else 'None'}...")
                logger.info(f"{'='*50}\n")

                # 将图片转换为base64
                base64_img, _  = image_to_base64(item['image_path'])

                # 构建提示词模板
                context_prompt = ""
                if prev_text and next_text:
                    context_prompt = f"""
        你是一位科研论文图像理解专家。请基于论文上下文和图片内容，生成该图片的英文语义描述。

        【论文上下文】
        前文：{prev_text}

        后文：{next_text}

        【任务要求】
        这是一篇科研论文中的图片，请：
        1. **优先参考上下文**：仔细阅读前后文，提取与图片相关的关键信息（如图片标题、图注、实验说明、数据含义等）
        2. **结合图片内容**：观察图片实际展示的内容（图表类型、坐标轴、数据趋势、架构组成等）
        3. **生成语义描述**：将上下文信息与图片内容融合，生成一段完整、准确的描述，使读者无需看图也能理解其含义
        4. **重点说明**：
        - 如果上下文提到了图号、图题，请包含
        - 如果是数据图表，说明表达的数据含义和趋势
        - 如果是架构图/流程图，说明其展示的系统或流程
        - 如果是实验场景图，说明实验环境和关键要素
        5. 描述长度控制在200-400字

        请直接给出描述，不要有"这张图片..."等前缀。
                            """
                elif prev_text:
                    context_prompt = f"""
        你是一位科研论文图像理解专家。请基于论文上下文和图片内容，生成该图片的英文语义描述。

        【论文上下文（前文）】
        {prev_text}

        【任务要求】
        这是一篇科研论文中的图片，请：
        1. **优先参考前文**：仔细阅读前文，提取与图片相关的关键信息（如图片标题、图注、实验说明等）
        2. **结合图片内容**：观察图片实际展示的内容
        3. **生成语义描述**：将上下文信息与图片内容融合，生成一段完整、准确的描述
        4. **重点说明**：图号、图题、数据含义、架构组成或实验要素
        5. 描述长度控制在200-400字

        请直接给出描述，不要有"这张图片..."等前缀。
                            """
                elif next_text:
                    context_prompt = f"""
        你是一位科研论文图像理解专家。请基于论文上下文和图片内容，生成该图片的英文语义描述。

        【论文上下文（后文）】
        {next_text}

        【任务要求】
        这是一篇科研论文中的图片，请：
        1. **优先参考后文**：仔细阅读后文，提取与图片相关的关键信息（如图片说明、结果分析等）
        2. **结合图片内容**：观察图片实际展示的内容
        3. **生成语义描述**：将上下文信息与图片内容融合，生成一段完整、准确的描述
        4. **重点说明**：图号、图题、数据含义、架构组成或实验要素
        5. 描述长度控制在200-400字

        请直接给出描述，不要有"这张图片..."等前缀。
                            """
                else:
                    context_prompt = """
        你是一位科研论文图像理解专家。请观察这张图片并生成英文描述。

        【任务要求】
        这是一篇科研论文中的图片，请：
        1. 识别图片类型（数据图表、架构图、流程图、实验场景图等）
        2. 描述图片展示的核心内容和关键信息
        3. 如果是图表，说明坐标轴、数据趋势等
        4. 如果是架构/流程图，说明主要组成部分
        5. 描述长度控制在200-400字

        请直接给出描述，不要有"这张图片..."等前缀。
                            """

                # 构建多模态消息
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": context_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"{base64_img}"
                            }
                        }
                    ]
                )

                # 调用模型生成描述  修改原始类型为图片的text字段
                response = qwen3_max.invoke([message])
                item['text'] = response.content
        
        return data_list

    def do_save_to_milvus(self, processed_data: List[Document]):
        """
        第一步：
        把Splitter之后的的数据（document对象列表），先转换为字典；
        第二步：
        把字典中的文本 和图片 ，进行向量化，然后再存入字典。
        第三步：
        最后写入向量数据库
        :param processed_data:
        :return:
        """
        # 第一步
        expanded_data = MilvusVectorSave.generate_image_description(MilvusVectorSave.doc_to_dict(processed_data))
        processed_data: List[Dict] = []
        # 处理每个 item
        for idx, item in enumerate(expanded_data, 1):
            # 限速控制
            limiter.acquire()

            # 处理 + 可选 429 重试
            if RETRY_ON_429:
                attempts = 0
                while True:
                    # 包装版处理
                    result = process_item_with_guard(item.copy())
                    # 检查是否成功
                    if result.get("text_content_dense"):
                        processed_data.append(result)
                        break
                    attempts += 1
                    if attempts > MAX_429_RETRIES:
                        print(f"[429重试] 超过最大重试次数，跳过 idx={idx}")
                        processed_data.append(result)
                        break
                    backoff = BASE_BACKOFF * (2 ** (attempts - 1)) * (0.8 + random.random() * 0.4)
                    print(f"[429重试] 第{attempts}次，sleep {backoff:.2f}s …")
                    time.sleep(backoff)
            else:
                # 若关闭 429 重试，这里同样使用包装版
                processed_data.append(process_item_with_guard(item.copy()))

            # 进度打印
            if idx % 20 == 0:
                print(f"[进度] 已处理 {idx}/{len(expanded_data)}")

        # 打印处理后的 item 内容
        # for item in processed_data:
        #     print(json.dumps(item, ensure_ascii=False, indent=4))
        
        # 第三步：写入向量数据库
        self.write_to_milvus(processed_data)
        
        # 返回处理后的数据
        return processed_data


# 便捷函数：用于在其他模块中调用
def do_save_to_milvus(processed_data: List[Document]) -> List[Dict]:
    """
    便捷函数：保存文档到 Milvus
    
    Args:
        processed_data: 处理后的文档列表
        
    Returns:
        保存到 Milvus 的数据列表
    """
    milvus_save = MilvusVectorSave()
    return milvus_save.do_save_to_milvus(processed_data)


if __name__ == "__main__":

    # ==================== 1. 创建 Milvus 集合 ====================
    milvus_vector_save = MilvusVectorSave()
    milvus_vector_save.create_collection(is_first=False)  # 首次运行设置为 True
    
    # 查看集合信息（调试用）
    # client = MilvusClient(uri=MILVUS_URI, user='root', password='Milvus')
    # res = client.describe_collection(collection_name=COLLECTION_NAME)
    # print("集合信息:")
    # print(res)
    
    # ==================== 2. 选择 OCR 方案 ====================
    
    # -------------------- 方案A：DotsOCR（图片+文本，图片有上下文）--------------------
    # 适用场景：需要保留图片，且图片需要结合上下文生成语义描述
    # 
    # DotsOCR 特点：
    #   ✅ MD 文件中包含 Base64 图片（图片位置在文档流中的正确位置）
    #   ✅ 图片前后有文本上下文，AI 可以结合上下文生成准确的图片描述
    #   ✅ 适合学术论文、技术文档（图表需要配合文字说明理解）
    #   ❌ Base64 图片质量可能较低
    #
    # 使用示例：
    # md_dir = r"F:\workspace\langgraph_project\Multimodal_RAG\output\GPT4技术报告"
    # splitter = MarkdownDirSplitter(
    #     images_output_dir=r"F:\workspace\langgraph_project\Multimodal_RAG\output\images"  # Base64 图片解码后保存位置
    # )
    # docs = splitter.process_md_dir(md_dir, source_filename="GPT4技术报告.pdf")
    # 
    # 生成的 Documents 结构：
    #   [文本块1] → [文本块2] → [图片1（有前后文）] → [文本块3] → [图片2（有前后文）] → ...
    # 
    # res: List[Dict] = milvus_vector_save.do_save_to_milvus(docs)
    
    # -------------------- 方案B：PaddleOCR（纯文本，不含图片）--------------------
    # 适用场景：只需要文本内容（包括公式、表格），不需要保留图片
    # 
    # PaddleOCR 特点：
    #   ✅ OCR 识别精度高（文字、公式、表格）
    #   ✅ 公式转 LaTeX，表格转 Markdown
    #   ✅ 生成纯文本 MD 文件（不包含图片）
    #   ✅ 文件小，处理快，向量化效果好
    #   ❌ 不保留原始图片（图片中的文字会被识别为文本）
    #
    # 使用示例（当前激活）：
    md_dir = r"F:\workspace\langgraph_project\Multimodal_RAG\paddle_ocr_output\大论文_多智能体系统主动容错控制及其在无人机中的应用"
    splitter = MarkdownDirSplitter(
        images_output_dir=r"F:\workspace\langgraph_project\Multimodal_RAG\paddle_ocr_output\images"  # PaddleOCR 的 MD 没有 Base64 图片，此参数无效
    )
    docs = splitter.process_md_dir(md_dir, source_filename="多智能体系统主动容错控制及其在无人机中的应用.pdf")
    # 
    # 生成的 Documents 结构：
    #   [文本块1] → [文本块2] → [文本块3] → ...（全是文本，没有图片 Documents）
    
    # ==================== 3. 存入 Milvus ====================
    # 自动完成以下步骤：
    #   1. Documents → 字典转换（doc_to_dict）
    #   2. 图片描述生成（generate_image_description，仅针对 category='image' 的条目，PaddleOCR 无图片会跳过）
    #   3. 向量化（text_content_dense + BM25 sparse）
    #   4. 批量写入 Milvus
    res: List[Dict] = milvus_vector_save.do_save_to_milvus(docs)

    # 打印
    # 打印关键数据
    for i, item in enumerate(res):
        print(f"\n==== 第{i+1}条数据 ====")
        # 打印文本内容前30字
        text = item.get('text', '')
        print(f"内容: {text[:30]}{'...' if len(text) > 30 else ''}")
        # 打印标题
        print(f"标题: {item.get('title', '')}")
        # 打印文件名、文件类型
        print(f"文件名: {item.get('filename', '')}")
        print(f"文件类型: {item.get('filetype', '')}")

