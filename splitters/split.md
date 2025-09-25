# MarkdownDirSplitter 架构文档

## 1. 概述

`MarkdownDirSplitter` 是一个高级的Markdown文档处理和分割工具，专为多模态RAG（检索增强生成）系统设计。该工具能够智能处理包含文本和图像的Markdown文档，实现结构化分割、图像提取和语义分块。

## 2. 核心功能

### 2.1 主要特性
- **多模态处理**: 同时处理文本内容和Base64编码的图像
- **结构化分割**: 基于Markdown标题层级进行文档分割
- **语义分块**: 对长文本进行智能语义切分
- **图像提取**: 提取并保存Markdown中的Base64图像为本地文件
- **标题层级追踪**: 维护完整的文档结构层级关系
- **批量处理**: 支持整个目录的Markdown文件批量处理

### 2.2 处理流程
1. **文档读取**: 按顺序读取目录中的所有Markdown文件
2. **标题分割**: 根据#、##、###标题层级进行初步分割
3. **图像处理**: 识别、提取并保存Base64图像，生成图像Document
4. **文本清理**: 移除图像标记，保留纯文本内容
5. **语义分块**: 对超长文本进行语义切分
6. **标题溯源**: 为每个文档块补充完整的标题层级信息

## 3. 架构设计

### 3.1 核心组件

```
MarkdownDirSplitter
├── 初始化配置
│   ├── images_output_dir: 图像输出目录
│   ├── text_chunk_size: 文本分块阈值
│   ├── headers_to_split_on: 标题分割规则
│   ├── text_splitter: Markdown标题切割器
│   └── semantic_splitter: 语义切割器
│
├── 图像处理模块
│   ├── save_base64_to_Image(): Base64图像解码保存
│   ├── process_images(): 处理Markdown中的Base64图像
│   └── remove_base64_images(): 移除Base64图像标记
│
├── 文档处理模块
│   ├── process_md_file(): 单个Markdown文件处理
│   ├── process_md_dir(): 目录批量处理
│   └── add_title_hierarchy(): 标题层级补充
│
└── 输出模块
    └── Document列表 (包含文本和图像Document)
```

### 3.2 依赖组件
- **LangChain**: `MarkdownHeaderTextSplitter`、`SemanticChunker`
- **PIL**: 图像处理
- **正则表达式**: Base64图像模式匹配
- **自定义工具**: `qwen_embeddings`、`get_sorted_md_files`

## 4. 数据结构

### 4.1 输入数据
- Markdown文件目录路径
- 图像输出目录路径
- 文本分块大小配置
- 源文件名（用于溯源）

### 4.2 输出数据结构

#### 文本Document结构
```python
Document(
    page_content="文本内容",
    metadata={
        "source": "原始文件名.pdf",
        "Header 1": "一级标题",
        "Header 2": "二级标题", 
        "Header 3": "三级标题",
        "embedding_type": "text"
    }
)
```

#### 图像Document结构
```python
Document(
    page_content="/path/to/images/hash.png",
    metadata={
        "source": "原始文件名.pdf",
        "alt_text": "图片",
        "embedding_type": "image"
    }
)
```

## 5. 关键算法

### 5.1 图像处理算法
```python
# Base64图像识别正则表达式
pattern = r'data:image/(.*?);base64,(.*?)\)'

# 图像哈希命名策略
hash_key = hashlib.md5(base64_data.encode()).hexdigest()
filename = f"{hash_key}.{img_type}"
```

### 5.2 文档分割策略
- **第一层**: 基于Markdown标题结构分割（#、##、###）
- **第二层**: 对超过阈值的文本块进行语义分割
- **标题继承**: 自动补充缺失的上级标题信息

### 5.3 文件排序算法
```python
# 基于页码的自然排序
def sort_key(file_path: str) -> int:
    match = re.search(r'_page_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')
```

## 6. 使用场景

### 6.1 适用场景
- **学术论文处理**: 处理包含图表的学术PDF转换后的Markdown
- **技术文档RAG**: 构建包含图像的技术文档知识库
- **多模态检索**: 需要同时检索文本和图像的应用
- **文档结构化**: 保持原文档标题层级结构的场景

### 6.2 典型工作流
```
PDF文档 → OCR转换 → Markdown文件 → MarkdownDirSplitter → 结构化Document → RAG系统
```

## 7. 配置参数

### 7.1 核心参数
- `images_output_dir`: 图像保存目录路径
- `text_chunk_size`: 文本分块大小阈值（默认1000字符）
- `headers_to_split_on`: 标题分割层级配置

### 7.2 分割器配置
- **MarkdownHeaderTextSplitter**: 基于标题的结构化分割
- **SemanticChunker**: 基于语义的智能分块
- **breakpoint_threshold_type**: 语义分割阈值类型

## 8. 扩展性

### 8.1 可扩展功能
- 支持更多图像格式处理
- 自定义标题层级配置
- API模式图像处理（直接处理Base64）
- 图像Alt文本智能识别
- 自定义语义分割策略

### 8.2 优化方向
- 并行处理提升性能
- 内存优化处理大文件
- 错误恢复和断点续传
- 图像质量和大小优化

## 9. 注意事项

### 9.1 限制条件
- 依赖特定的Markdown格式（标题层级）
- Base64图像需要标准格式
- 文件命名需遵循特定模式（_page_X）

### 9.2 最佳实践
- 合理设置text_chunk_size以平衡检索效果
- 确保图像输出目录有足够存储空间
- 定期清理临时图像文件
- 监控内存使用情况处理大量文件

## 10. 示例用法

```python
# 初始化分割器
splitter = MarkdownDirSplitter(
    images_output_dir="./output/images",
    text_chunk_size=1000
)

# 处理Markdown目录
docs = splitter.process_md_dir(
    md_dir="./output/chapter1", 
    source_filename="原始文档.pdf"
)

# 分析输出结果
for doc in docs:
    if doc.metadata['embedding_type'] == 'text':
        print(f"文本块: {doc.page_content[:50]}...")
    else:
        print(f"图像块: {doc.page_content}")
```

此架构设计确保了多模态文档的高效处理，为RAG系统提供了结构化、语义丰富的文档块，支持文本和图像的统一检索和处理。