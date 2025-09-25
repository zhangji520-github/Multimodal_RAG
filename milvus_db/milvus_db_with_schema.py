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
from utils.embeddings_utils import build_work_items, process_item_with_guard, limiter, RETRY_ON_429, MAX_429_RETRIES, BASE_BACKOFF
from langchain_milvus import Milvus
from pymilvus import DataType, Function, FunctionType, MilvusClient
from utils.embeddings_utils import WorkItem
from env_utils import COLLECTION_NAME, MILVUS_URI
import logging
import time
import random



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
        schema.add_field("text", DataType.VARCHAR, max_length=6000, enable_analyzer=True,
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
                    doc_dict['text'] = doc_dict['title'] + '：' + doc.page_content
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
        
        try:
            insert_res = self.client.insert(collection_name=COLLECTION_NAME, data=processed_data)
            print(f"[Milvus] 成功写入 {len(processed_data)} 条数据.IDs 示例: {insert_res['ids'][:5]}")
        except Exception as e:
            logger.error(f"🐶写入Milvus失败: {e}")
            raise e
        
    def do_save_to_milvus(self, docs: List[Document]):
        """
        将 LangChain 的 Document 对象列表（来自文本/图像分割器）
        → 转换为结构化字典
        → 调用 DashScope 多模态 API 生成统一向量（dense）
        → 写入 Milvus 向量数据库 
        返回的 result 是增强后的字典，必定包含 dense 字段：
            成功 dense = [0.1, -0.2, ...]
            失败 dense = []
            图像任务还会设置 text = "图片"（便于前端展示）
        """
        # 第一步：把 Document 转换为结构化字典
        expanded_data = self.doc_to_dict(docs)
        # 第二步：调用 DashScope 多模态 API 生成统一向量（dense）
        work_items: List[WorkItem] = build_work_items(expanded_data)
        
        processed_data: List[Dict] = []
        for idx,wi in enumerate(work_items, start=1):
            limiter.acquire()

            # 情况一：启用 429 限流重试（RETRY_ON_429 = True）
            if RETRY_ON_429:
                attempts = 0
                while True:
                    res = process_item_with_guard(wi.item.copy(), wi.mode, wi.api_image)
                    if res.get('text_content_dense'):
                        processed_data.append(res)
                        break
                    attempts += 1
                    if attempts >= MAX_429_RETRIES:
                        print(f"🐶429重试次数超过最大值: {MAX_429_RETRIES}")
                        processed_data.append(res)
                        break
                    backoff = BASE_BACKOFF * (2 ** (attempts - 1)) * (0.8 + random.random() * 0.4)
                    print(f"[429重试] 第{attempts}次，sleep {backoff:.2f}s …")
                    time.sleep(backoff)
            # 情况二：不启用 429 限流重试（RETRY_ON_429 = False） 适用于调试或低频场景
            else:
                result = process_item_with_guard(wi.item.copy(), mode=wi.mode, api_image=wi.api_image)
                processed_data.append(result)
            # 进度提示 每 20 个 item 打印一次进度，避免日志刷屏
            if idx % 20 == 0:
                print(f"[进度] 已处理 {idx}/{len(work_items)}")

        # 第三步
        self.write_to_milvus(processed_data)
        return processed_data

if __name__ == "__main__":

    # 创建表结构
    milvus_vector_save = MilvusVectorSave()
    milvus_vector_save.create_collection(is_first=True)
    
    # 查看集合信息
    # client = MilvusClient(uri=MILVUS_URI, user='root', password='Milvus')
    # res = client.describe_collection(collection_name=COLLECTION_NAME)
    # print("集合信息:")
    # print(res)
    md_dir = r"F:\workspace\langgraph_project\Multimodal_RAG\output\RBF神经网络无人艇包含控制推导"
    splitter = MarkdownDirSplitter(images_output_dir=r"F:\workspace\langgraph_project\Multimodal_RAG\output\images")
    docs = splitter.process_md_dir(md_dir, source_filename="RBF神经网络无人艇包含控制推导.pdf")

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

