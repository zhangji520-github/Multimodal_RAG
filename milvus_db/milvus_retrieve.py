from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, RRFRanker
from typing import List, Dict, Optional
import sys
import os

# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.embeddings_utils import call_dashscope_once
from env_utils import COLLECTION_NAME


class MilvusRetriever:
    def __init__(self, collection_name: str = COLLECTION_NAME, milvus_client: MilvusClient = None, top_k: int = 5):
        self.collection_name = collection_name
        self.client: MilvusClient = milvus_client
        self.top_k = top_k


    def dense_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        稠密向量检索（支持文本和图像的语义搜索）
        :param query_embedding: 已经向量化的内容 1024维向量
        :param limit: 返回结果数量
        :return: 检索结果列表
        """
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="text_content_dense",  # 使用正确的稠密向量字段名
            limit=limit,
            output_fields=["text", "category", "filename", "image_path", "title"],
            search_params=search_params,
        )
        return res[0] if res else []

    def sparse_search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """
        内容稀疏向量搜索（BM25全文检索）
        :param query: 搜索的关键词文本
        :param limit: 返回结果数量
        :return: 检索结果列表
        """
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="text_content_sparse",  # 使用正确的内容稀疏向量字段名
            limit=limit,
            output_fields=["text", "category", "filename", "image_path", "title"],
            search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
        )
        return res[0] if res else []

    def sparse_search_title(self, query: str, limit: int = 5) -> List[Dict]:
        """
        标题稀疏向量搜索（BM25标题检索）
        :param query: 搜索的关键词文本
        :param limit: 返回结果数量
        :return: 检索结果列表
        """
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="title_sparse",  # 使用标题稀疏向量字段
            limit=limit,
            output_fields=["text", "category", "filename", "image_path", "title"],
            search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
        )
        return res[0] if res else []

    def get_query_embedding(self, query: str, mode: str = "text") -> List[float]:
        """
        将查询文本转换为向量嵌入
        :param query: 查询文本
        :param mode: 模式，'text' 或 'image'
        :return: 向量嵌入列表
        """
        if mode == "text":
            input_data = [{"text": query}]
        else:
            input_data = [{"image": query}]  # query为图像路径或base64
        
        ok, embedding, _, _ = call_dashscope_once(input_data)
        return embedding if ok else []

    def hybrid_search(self, 
                     query: str, 
                     weights: Optional[List[float]] = None,
                     ranker_type: str = "rrf",
                     rrf_k: int = 60,
                     limit: int = 10) -> List[Dict]:
        """
        官方混合检索：使用 Milvus 原生 hybrid_search API
        :param query: 查询文本
        :param weights: 权重列表 [dense_weight, sparse_content_weight, sparse_title_weight]
        :param ranker_type: 重排序策略 "rrf" 或 "weighted"
        :param rrf_k: RRF 算法的 k 参数，默认 60
        :param limit: 返回结果数量
        :return: 混合检索结果列表
        """
        # 默认权重配置
        if weights is None:
            weights = [0.6, 0.3, 0.1]  # [dense, sparse_content, sparse_title]
        
        # 1. 获取查询向量
        query_embedding = self.get_query_embedding(query, mode="text")
        if not query_embedding:
            print("⚠️ 获取查询向量失败，仅使用稀疏向量检索")
            return self.sparse_search_content(query, limit=limit)
        
        # 2. 构建搜索请求列表
        search_requests = []
        
        # 稠密向量搜索请求
        dense_search = AnnSearchRequest(
            data=[query_embedding],
            anns_field="text_content_dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit
        )
        search_requests.append(dense_search)
        
        # 内容稀疏向量搜索请求
        sparse_content_search = AnnSearchRequest(
            data=[query],
            anns_field="text_content_sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=limit
        )
        search_requests.append(sparse_content_search)
        
        # 标题稀疏向量搜索请求
        sparse_title_search = AnnSearchRequest(
            data=[query],
            anns_field="title_sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=limit
        )
        search_requests.append(sparse_title_search)
        
        # 3. 选择重排序策略
        if ranker_type == "weighted":
            ranker = WeightedRanker(*weights)
        else:  # rrf
            ranker = RRFRanker(k=rrf_k)
        
        # 4. 执行混合搜索
        try:
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=search_requests,
                ranker=ranker,
                limit=limit,
                output_fields=["text", "category", "filename", "image_path", "title"]
            )
            return results[0] if results else []
        except Exception as e:
            print(f"❌ 混合搜索失败: {e}")
            # 降级到单一检索
            return self.dense_search(query_embedding, limit=limit)

    def hybrid_search_with_weighted_ranker(self, 
                                          query: str, 
                                          weights: List[float] = None,
                                          limit: int = 10) -> List[Dict]:
        """
        使用加权重排序器的混合检索
        :param query: 查询文本
        :param weights: 权重列表 [dense, sparse_content, sparse_title]
        :param limit: 返回结果数量
        :return: 检索结果列表
        """
        if weights is None:
            weights = [0.6, 0.3, 0.1]
        
        return self.hybrid_search(
            query=query,
            weights=weights,
            ranker_type="weighted",
            limit=limit
        )
    
    def hybrid_search_with_rrf_ranker(self, 
                                     query: str, 
                                     k: int = 60,
                                     limit: int = 10) -> List[Dict]:
        """
        使用 RRF 重排序器的混合检索
        :param query: 查询文本
        :param k: RRF 算法的 k 参数
        :param limit: 返回结果数量
        :return: 检索结果列表
        """
        return self.hybrid_search(
            query=query,
            ranker_type="rrf",
            rrf_k=k,
            limit=limit
        )

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        语义搜索（仅使用稠密向量）
        :param query: 查询文本
        :param limit: 返回结果数量
        :return: 检索结果列表
        """
        query_embedding = self.get_query_embedding(query, mode="text")
        if not query_embedding:
            return []
        
        return self.dense_search(query_embedding, limit=limit)

    def keyword_search(self, query: str, limit: int = 5, search_title: bool = True) -> List[Dict]:
        """
        关键词搜索（仅使用稀疏向量）
        :param query: 查询文本
        :param limit: 返回结果数量
        :param search_title: 是否同时搜索标题
        :return: 检索结果列表
        """
        content_results = self.sparse_search_content(query, limit=limit)
        
        if search_title:
            title_results = self.sparse_search_title(query, limit=limit)
            # 简单合并去重
            seen_ids = set()
            combined_results = []
            
            for item in content_results + title_results:
                item_id = f"{item.get('entity', {}).get('filename', '')}__{item.get('entity', {}).get('text', '')[:50]}"
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    combined_results.append(item)
            
            return combined_results[:limit]
        
        return content_results


# 使用示例
if __name__ == "__main__":
    from env_utils import MILVUS_URI
    
    # 初始化检索器
    client = MilvusClient(uri=MILVUS_URI, user='root', password='Milvus')
    retriever = MilvusRetriever(collection_name=COLLECTION_NAME, milvus_client=client)
    
    # 测试查询
    test_query = "神经网络"
    
    print("=" * 60)
    print(f"查询: {test_query}")
    print("=" * 60)
    
    # 1. 官方混合检索 - RRF 重排序（推荐）
    print("\n🔍 混合检索 (RRF重排序):")
    rrf_results = retriever.hybrid_search_with_rrf_ranker(
        query=test_query,
        k=60,
        limit=5
    )
    
    for i, result in enumerate(rrf_results, 1):
        entity = result.get('entity', {})
        distance = result.get('distance', 0)
        print(f"{i}. 标题: {entity.get('title', '')}")
        print(f"   内容: {entity.get('text', '')[:100]}...")
        print(f"   文件: {entity.get('filename', '')}")
        print(f"   类型: {entity.get('category', '')}")
        print(f"   相似度: {distance:.4f}")
        print()
    
    # 2. 混合检索 - 加权重排序
    print("\n⚖️ 混合检索 (加权重排序):")
    weighted_results = retriever.hybrid_search_with_weighted_ranker(
        query=test_query,
        weights=[0.6, 0.3, 0.1],  # [dense, sparse_content, sparse_title]
        limit=3
    )
    
    # for i, result in enumerate(weighted_results, 1):
    #     entity = result.get('entity', {})
    #     distance = result.get('distance', 0)
    #     print(f"{i}. {entity.get('title', '')} - 相似度: {distance:.4f}")
    #     print(f"   内容: {entity.get('text', '')[:50]}...")
    
    # # 3. 语义搜索
    # print("\n🎯 语义搜索结果:")
    # semantic_results = retriever.semantic_search(test_query, limit=3)
    # for i, result in enumerate(semantic_results, 1):
    #     entity = result.get('entity', {})
    #     distance = result.get('distance', 0)
    #     print(f"{i}. {entity.get('title', '')} - 相似度: {distance:.4f}")
    
    # # 4. 关键词搜索
    # print("\n🔤 关键词搜索结果:")
    # keyword_results = retriever.keyword_search(test_query, limit=3)
    # for i, result in enumerate(keyword_results, 1):
    #     entity = result.get('entity', {})
    #     distance = result.get('distance', 0)
    #     print(f"{i}. {entity.get('title', '')} - BM25分数: {distance:.4f}")
    
    # # 5. 对比不同重排序策略
    # print("\n📊 不同重排序策略对比:")
    # print("RRF 策略结果数量:", len(rrf_results))
    # print("加权策略结果数量:", len(weighted_results))
    
    # # 6. 自定义权重测试
    # print("\n🎛️ 自定义权重测试 (专注于稀疏向量):")
    # custom_results = retriever.hybrid_search_with_weighted_ranker(
    #     query=test_query,
    #     weights=[0.2, 0.6, 0.2],  # 更关注关键词匹配
    #     limit=3
    # )
    
    # for i, result in enumerate(custom_results, 1):
    #     entity = result.get('entity', {})
    #     print(f"{i}. {entity.get('title', '')}")