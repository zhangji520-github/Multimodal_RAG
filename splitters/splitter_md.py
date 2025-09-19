import os
import base64
from PIL import Image
import io
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from llm_utils import qwen_embeddings
from langchain_core.documents import Document
import re
import hashlib
from typing import List
from utils.common_utils import get_sorted_md_files
from utils.log_utils import log





class MarkdownDirSplitter:

    def __init__(self, images_output_dir: str, text_chunk_size: int = 1000):
        """
        :params images_output_dir: 用于保存从 Markdown 中提取的 Base64 图片的本地目录。如果调用api可以直接对base64进行处理，如果私有化部署的gem则需要对jpg处理
        :params text_chunk_size:  文本分块阈值，默认 1000 字符。超过该长度的文本块会进一步语义分割。
        """
        self.images_output_dir = images_output_dir
        self.text_chunk_size = text_chunk_size
        os.makedirs(self.images_output_dir, exist_ok=True)

        # 定义标题切割层级
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),   
        ]

        # 初始化 Markdown 标题切割器
        self.text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

        # 初始化语义切割器
        self.semantic_splitter = SemanticChunker(
            qwen_embeddings, 
            breakpoint_threshold_type="percentile"
        )

    def save_base64_to_Image(self, base64_str: str, output_path: str ) -> None:
        '''将 Base64 编码的图片数据解码并保存为图像文件 因为私有化部署的gem模型无法直接处理base64'''
        try:
            if base64_str.startswith("data:image"):
                base64_str = base64_str.split(",", 1)[1]           # Data URL 格式 的 Base64 图像字符串

            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img.save(output_path)
        except Exception as e:
            print(f"保存图片失败: {e}")
            raise  

    # 处理Markdown中的base64图片 私有化的话你必须换成jpg或者png
    def process_images(self, content: str, source: str) -> List[Document]:
        """
        处理Markdown中的base64图片
        :param content:  Markdown 内容（字符串）就是警告md切割器之后的page_content
        :param source:  source 是当前正在处理的 Markdown 文件的路径（文件名或完整路径）
        
        """
        image_docs = []
        pattern = r'data:image/(.*?);base64,(.*?)\)'  # 正则匹配base64图片

        def replace_image(match):
            img_type = match.group(1).split(';')[0]  # match.group(1) → 如 "png" 或 "jpeg; charset=utf-8"，用 .split(';')[0] 取第一部分 → 得到干净的格式名。
            base64_data = match.group(2)         #  match.group(2)   Base64 编码的图像数据（不包含前缀）

            # 生成唯一文件名
            hash_key = hashlib.md5(base64_data.encode()).hexdigest()
            filename = f"{hash_key}.{img_type if img_type in ['png', 'jpg', 'jpeg'] else 'png'}"   # 根据 img_type 设置扩展名，只允许 png/jpg/jpeg，其他默认用 png（安全兜底）
            img_path = os.path.join(self.images_output_dir, filename) # 组合完整路径：images_output_dir/hash.png

            # 保存图片
            self.save_base64_to_Image(base64_data, img_path)

            # 创建图片Document
            image_docs.append(Document(
                page_content=str(img_path),
                metadata={
                    "source": source,
                    "alt_text": "图片",
                    "embedding_type": "image"
                }
            ))

            return "[图片]"

        # 替换所有base64图片 
        content = re.sub(pattern, replace_image, content, flags=re.DOTALL)
        return image_docs      # 返回所有被提取图片对应的 Document 列表，供后续与文本块合并使用。
    '''
    通过 embedding_type 字段区分，系统可分别处理
    处理之后的图片专属 Document 结构, 与文本Document结构平起平坐哈哈:
    Document(
        page_content = "/path/to/images/abc123.png",  # ← 图片在本地磁盘的路径
        metadata = {
            "source": "xxx.md",          # ← 来源文件
            "alt_text": "图片",           # ← 替代文本（当前固定，可扩展）
            "embedding_type": "image"    # ← 标记这是图片块，非文本 
        }
    )
    '''
    def process_image_with_api():
        '''使用API处理图片,返回图片的Document列表'''
        pass

    def remove_base64_images(self, text: str) -> str:
        """移除所有Base64图片标记"""
        pattern = r'!\[\]\(data:image/(.*?);base64,(.*?)\)'
        return re.sub(pattern, '', text)

    def process_md_file(self, md_file: str) -> List[Document]:
        """
        读取一个 Markdown 文件 → 按标题结构分割 → 提取并保存其中的 Base64 图片 → 清洗文本 → 对长文本进行语义分块 → 最终返回结构化、可嵌入的 Document 列表（包含文本块和图片路径块）
        :params md_file —— Markdown 文件的路径（字符串）
        """
        with open(md_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # 1. 按标题结构分割
        split_documents: List[Document] = self.text_splitter.split_text(content)
        documents = []
        for doc in split_documents:
            # 2. 提取并保存其中的 Base64 图片
            if 'data:image/' in doc.page_content:
                # 拿到图片的专属document
                image_docs: List[Document] = self.process_images(doc.page_content, md_file)
                # 移除图片之后，# 如果清洗后还有文本内容，创建一个纯文本 Document 并标记 embedding_type='text' 
                # # 剩余的文本内容，给上doc.metadata['embedding_type'] = 'text'作为文本的embedding_type
                cleaned_content = self.remove_base64_images(doc.page_content)
                if cleaned_content.strip():
                    doc.metadata['embedding_type'] = 'text'
                    documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))
                documents.extend(image_docs)       # 将处理好的图片文档添加到结果列表
            else:
                doc.metadata['embedding_type'] = 'text'
                documents.append(doc)

        # 语义分割
        final_docs = []
        for d in documents:
            if len(d.page_content) > self.text_chunk_size:
                final_docs.extend(self.semantic_splitter.split_documents([d]))
            else:
                final_docs.append(d)

        # 添加标题层级 用于后续的查询和检索
        return final_docs
    
    def add_title_hierarchy(self, documents: List[Document], source_filename: str) -> List[Document]:
        """为文档添加标题层级结构"""
        current_titles = {1: "", 2: "", 3: ""}
        processed_docs = []

        for doc in documents:
            new_metadata = doc.metadata.copy()
            new_metadata['source'] = source_filename

            # 更新标题状态
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key in new_metadata:
                    current_titles[level] = new_metadata[header_key]
                    for lower_level in range(level + 1, 4):
                        current_titles[lower_level] = ""

            # 补充缺失的标题
            for level in range(1, 4):
                header_key = f'Header {level}'
                if header_key not in new_metadata:
                    new_metadata[header_key] = current_titles[level]
                elif current_titles[level] != new_metadata[header_key]:
                    new_metadata[header_key] = current_titles[level]

            processed_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata
                )
            )

        return processed_docs

    def process_md_dir(self, md_dir: str, source_filename: str) -> List[Document]:
        """
        处理一个目录下的所有 Markdown 文件，并将它们统一结构化、分块、添加标题层级后，返回一个完整的 Document 列表
        :param md_dir: Markdown 文件所在的目录路径 "./output/chapter1"，里面可能有 page_01.md, page_02.md 等
        :param source_filename:  md数据的原始文件（pdf）。
        :return:
        """
        # 读取 所有 .md 文件，按页码/章节排序
        md_files = get_sorted_md_files(md_dir)
        all_documents = []
        for md_file in md_files:
            log.info(f"真正处理的文件为： {md_file}")
            all_documents.extend(self.process_md_file(md_file))    
        """
        process_md_file 内部做了什么？
        读取文件内容
        按标题分割（#、##、###）→ 得到多个块（每个块带部分标题 metadata）
        提取 Base64 图片：
        保存到 images_output_dir
        生成图片 Document（page_content=图片路径，metadata.source=当前 md_file）
        原文中的图片替换为 [图片]
        语义分块：长文本按语义切分
            返回该文件的所有 Document 列表
        → 此时，每个 Document 的 metadata['source'] 是它所在的 .md 文件路径（如 "./converted_md/page_01.md"）
        """
        # 添加标题层级
        return self.add_title_hierarchy(all_documents, source_filename)


if __name__ == "__main__":
    md_dir = "F:\workspace\langgraph_project\Multimodal_RAG\output\RBF神经网络无人艇包含控制推导"

    splitter = MarkdownDirSplitter(images_output_dir="F:\workspace\langgraph_project\Multimodal_RAG\output\images")

    docs = splitter.process_md_dir(md_dir, source_filename="RBF神经网络无人艇包含控制推导.pdf")

    for i, doc in enumerate(docs):
        print(f"\n文档{i+1}:")
        print(doc)
        print("-"*100)
        print(f"内容: {doc.page_content[:30]}")
        print(f"元数据: {doc.metadata}")
        print('-'*100)

        print(f"一级标题: {doc.metadata['Header 1']}")
        print(f"二级标题: {doc.metadata['Header 2']}")
        print(f"三级标题: {doc.metadata['Header 3']}")
        print('-'*100)

