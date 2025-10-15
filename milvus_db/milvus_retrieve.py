import sys
import os
# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, RRFRanker
from typing import List, Dict, Any
from utils.embeddings_utils import call_dashscope_once, image_to_base64
from milvus_db.milvus_db_with_schema import logger
from env_utils import COLLECTION_NAME, MILVUS_URI

class MilvusRetriever:
    def __init__(self, collection_name: str, milvus_client: MilvusClient, top_k: int = 3):
        self.collection_name = collection_name
        self.client: MilvusClient = milvus_client
        self.top_k = top_k

    def dense_search(self, query_embedding, limit=5):
        """
        密集向量检索
        :param query_embedding: 查询向量
        :param limit: 返回结果数量
        :return: 查询结果
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        res = self.client.search(
            collection_name=self.collection_name,
            data = [query_embedding],
            anns_field="text_content_dense",
            limit=limit,
            search_params=search_params,
            output_fields=["text", "category", "filename", "image_path", "title"],
        )
        logger.info(f"✅ 密集向量检索成功，返回 {len(res[0])} 条结果")
        return res[0]
    
    def sparse_content_search(self, query, limit=5):
        """
        内容稀疏向量检索
        :param query: 查询内容
        :param limit: 返回结果数量
        :return: 查询结果
        """
        search_params = {"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}
        res = self.client.search(
            collection_name=self.collection_name,
            data = [query],
            anns_field="text_content_sparse",
            limit=limit,
            search_params=search_params,
            output_fields=["text", "category", "filename", "image_path", "title"],
        )
        logger.info(f"✅ 内容稀疏向量检索成功，返回 {len(res[0])} 条结果")
        return res[0]
    
    def sparse_title_search(self, query, limit=5):
        """
        标题稀疏向量检索
        :param query: 查询标题
        :param limit: 返回结果数量
        :return: 查询结果
        """
        search_params = {"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}
        res = self.client.search(
            collection_name=self.collection_name,
            data = [query],
            anns_field="title_sparse",
            limit=limit,
            search_params=search_params,
            output_fields=["text", "category", "filename", "image_path", "title"],
        )
        logger.info(f"✅ 标题稀疏向量检索成功，返回 {len(res[0])} 条结果")
        return res
    
    def hybrid_search(
        self,
        query_dense_embedding,
        query_text,
        sparse_weight=1.0,
        dense_weight=1.0,
        limit=10
    ):
        """
        混合检索 都是针对"text_content_sparse"字段的检索,包括文本以及图片的dense向量 
        当前只支持单传文本或者图片
        :param query_dense_embedding: 查询密集向量 图片或者文本经过DashScope API 转化为的密集向量
        :param query_text: 原始的查询文本 
        :param sparse_weight: 稀疏向量权重 经过BM25算法转化为的稀疏向量
        :param dense_weight: 密集向量权重
        :param limit: 返回结果数量
        :return: 查询结果
        """
        # 每个 AnnSearchRequest 代表针对特定向量字段的基础 ANN 搜索请求 不管是图片还是文本，我们都可以统一转化为dense向量
        dense_search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        dense_req = AnnSearchRequest(
            data = [query_dense_embedding],
            anns_field = "text_content_dense",
            limit = limit,
            param = dense_search_params,
        )

        sparse_search_params = {"metric_type": "BM25", "params": {'drop_ratio_search': 0.2}}
        sparse_req = AnnSearchRequest(
            data = [query_text],
            anns_field = "text_content_sparse",
            limit = limit,
            param = sparse_search_params,
        )

        # 在混合搜索中，重排序是一个关键步骤，它整合了来自多个向量搜索的结果，以确保最终输出是最相关和最准确的
        ranker_weighted = WeightedRanker(sparse_weight, dense_weight)
        # ranker_rrf = RRFRanker(k=100)

        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs = [dense_req, sparse_req],
            ranker = ranker_weighted,
            limit = limit,
            output_fields = ["text", "category", "filename", "image_path", "title"],
        )[0]
        logger.info(f"🔍 混合检索成功，返回 {len(res)} 条结果 (dense权重={dense_weight}, sparse权重={sparse_weight})")
        return res

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        检索
        :param query: # 在我们的使用场景中，用户要么输入一段文字，要么输入一个本地图片的完整路径。这两种情况不会混淆 如果 query 这个字符串，在当前电脑上恰好是一个真实存在的文件路径，那我就当它是图片来处理；否则，我就当它是普通文本
        :return: 查询结果
        """
        if os.path.isfile(query):
            # 构建图像输入数据，满足DashScope API 的要求
            logger.info(f"📷 检测到图片查询: {query}")
            # image_to_base64 返回 (api_img, img) 元组，我们只需要第一个元素
            base64_img, _ = image_to_base64(query)
            input_data = [{'image': base64_img}]
            ok, dense_embedding, status, retry_after = call_dashscope_once(input_data)  # 调用API获取图像嵌入向量   调用 DashScope 多模态 API 时，只需要纯 Base64 字符串，不需要 data:image/... 前缀。
        else:
            # 构建文本输入数据，满足DashScope API 的要求
            logger.info(f"📝 检测到文本查询: {query}")
            input_data = [{'text': query}]
            ok, dense_embedding, status, retry_after = call_dashscope_once(input_data)  # 调用API获取文本嵌入向量
        
        if ok:
            if os.path.isfile(query):   # 纯图片之可以使用dense_search
                results = self.dense_search(dense_embedding, limit=self.top_k)
            else:
                results = self.hybrid_search(dense_embedding, query, limit=self.top_k)
        else:
            raise ValueError(f"Failed to get dense embedding: {status}")
        
        # return results

        docs = []
        # print(results)
        for hit in results:
            docs.append({"text": hit.text, "category": hit.category, "filename": hit.filename, "image_path": hit.image_path, "title": hit.title})

        logger.info(f"🎉 检索完成！成功返回 {len(docs)} 条文档结果")
        return docs
    
if __name__ == "__main__":
    retrieve = MilvusRetriever(collection_name=COLLECTION_NAME, milvus_client=MilvusClient(uri=MILVUS_URI, user='root', password='Milvus'))
    docs = retrieve.retrieve("Internal factual eval by category")
    for doc in docs:
        print(doc) 