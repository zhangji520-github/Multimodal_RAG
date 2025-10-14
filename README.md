# Multimodal RAG

这是一个多模态检索增强生成 (Multimodal Retrieval-Augmented Generation) 项目。

## 项目简介

本项目旨在构建一个支持文本、图像等多种模态数据的检索增强生成系统。

## 功能特性

- 多模态数据处理
- 检索增强生成
- 基于 LangGraph 的工作流

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 多模态检索

`milvus_db/milvus_retrieve.py` 提供了统一的 `MilvusRetriever` 接口，支持三种检索模式：

1. **文本检索**：调用 `search_text("查询内容")` 或 `search(query_text="查询内容")`，默认启用稠密向量 + BM25 混合检索。
2. **图片检索**：调用 `search_image("图片路径或 URL")`，内部自动将本地图片转为 Base64 并走稠密向量召回。
3. **图文混合检索**：调用 `search_mixed("文本", "图片路径", text_weight=0.4)`，通过加权融合文本与图片向量完成召回。

快速示例：

```python
from milvus_db.milvus_retrieve import MilvusRetriever

retriever = MilvusRetriever()

# 文本
text_hits = retriever.search_text("神经网络控制", k=3)

# 图片
image_hits = retriever.search_image("output/images/demo.png", k=3)

# 图文混合
mixed_hits = retriever.search_mixed(
	text="自主导航",
	image_path="output/images/demo.png",
	k=3,
	text_weight=0.6,
)
```

> 注意：运行前需确保 Milvus 服务已启动，并正确配置 `MILVUS_URI`、DashScope API Key 等环境变量。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

MIT License
