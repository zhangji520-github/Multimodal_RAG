import os
import re
from pathlib import Path
from typing import List
import shutil

def get_filename(file_path, with_extension=True):
    """
    获取文件名
    :param file_path: 文件绝对路径
    :param with_extension: 是否包含扩展名
    :return: 文件名
    """
    if with_extension:
        return os.path.basename(file_path)
    else:
        return Path(file_path).stem


def get_sorted_md_files(input_dir: str) -> List[str]:
    """
    按照页号，把所有的md文件排序。（xx_0.md, xx_1.md, xx_2.md, .... xx_12.md）
    获取指定目录下所有 .md 文件（递归查找子目录），并按照数字排序
    支持两种文件名格式：
    - _page_X 格式（DotsOCR）
    - _X 格式（PaddleOCR）
    """
    # 递归获取所有 .md 文件（排除 _nohf.md）
    md_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.md') and not f.endswith('_nohf.md'):
                md_files.append(os.path.join(root, f))

    # 定义排序 key 函数：提取文件名中的数字
    def sort_key(file_path: str) -> int:
        filename = os.path.basename(file_path)
        # 优先匹配 _page_X 格式（DotsOCR）
        match = re.search(r'_page_(\d+)', filename)
        if match:
            return int(match.group(1))
        # 然后匹配 _X.md 格式（PaddleOCR）
        match = re.search(r'_(\d+)\.md$', filename)
        if match:
            return int(match.group(1))
        # 如果没有找到数字，则排在最后
        return float('inf')

    # 按照数字排序
    sorted_files = sorted(md_files, key=sort_key)
    return sorted_files

def get_surrounding_text_content(data_list, index, max_prev_chunks=3, max_next_chunks=2, max_chars=800):
    """
    获取指定图片字典的前后文本字典的文本内容（支持多段聚合）。
    
    对于科研论文，图片通常需要更多上下文来理解，因此会聚合多段前后文本。

    参数:
        data_list: 包含字典的列表，每个字典有'text'和'image_path'键
        index: 当前图片字典在列表中的索引
        max_prev_chunks: 最多向前查找的文本段数，默认3段
        max_next_chunks: 最多向后查找的文本段数，默认2段
        max_chars: 聚合文本的最大字符数，默认800字（避免上下文过长）

    返回:
        一个元组 (prev_text, next_text):
        - prev_text: 前面多个文本块聚合的内容，如果找不到则为 None
        - next_text: 后面多个文本块聚合的内容，如果找不到则为 None
    """
    prev_texts = []
    next_texts = []

    # 查找前面的文本字典（最多max_prev_chunks段）
    i = index - 1
    collected_prev = 0
    prev_char_count = 0
    while i >= 0 and collected_prev < max_prev_chunks:
        # 检查是否为文本字典：image_path为空字符串或None
        if 'text' in data_list[i] and not data_list[i].get('image_path'):
            text = data_list[i].get('text', '')
            if text:  # 只添加非空文本
                # 检查是否超过字符限制
                if prev_char_count + len(text) > max_chars and prev_texts:
                    break
                prev_texts.insert(0, text)  # 插入到开头，保持顺序
                prev_char_count += len(text)
                collected_prev += 1
        i -= 1

    # 查找后面的文本字典（最多max_next_chunks段）
    j = index + 1
    collected_next = 0
    next_char_count = 0
    while j < len(data_list) and collected_next < max_next_chunks:
        # 检查是否为文本字典：image_path为空字符串或None
        if 'text' in data_list[j] and not data_list[j].get('image_path'):
            text = data_list[j].get('text', '')
            if text:  # 只添加非空文本
                # 检查是否超过字符限制
                if next_char_count + len(text) > max_chars and next_texts:
                    break
                next_texts.append(text)
                next_char_count += len(text)
                collected_next += 1
        j += 1

    # 聚合文本，使用换行分隔
    prev_text = '\n'.join(prev_texts) if prev_texts else None
    next_text = '\n'.join(next_texts) if next_texts else None

    return prev_text, next_text

def delete_directory_if_non_empty(dir_path):
    """
    删除指定目录（如果该目录存在且非空）

    参数:
    dir_path (str): 要检查并可能删除的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"目录 '{dir_path}' 不存在，无需删除。")
        return False

    # 确认路径是一个目录
    if not os.path.isdir(dir_path):
        print(f"'{dir_path}' 不是一个目录。")
        return False

    # 检查目录是否为空
    if not os.listdir(dir_path):
        print(f"目录 '{dir_path}' 为空，无需删除。")
        return False

    # 目录存在且非空，执行删除操作
    try:
        shutil.rmtree(dir_path)
        print(f"成功删除非空目录: '{dir_path}'")
        return True
    except Exception as e:
        print(f"删除目录 '{dir_path}' 时发生错误: {e}")
        return False

if __name__ == '__main__':
    # 使用示例
    file_path = "/home/user/documents/example.txt"
    print(get_filename(file_path))          # 输出: example.txt
    print(get_filename(file_path, False))  # 输出: example

    # 使用示例
    # directory_to_delete = "E:\\my_project\\graph_rag_demo\\output\\demo_pdf1"
    # delete_directory_if_non_empty(directory_to_delete)