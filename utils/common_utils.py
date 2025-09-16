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
    获取指定目录下所有 .md 文件，并按照 _page_X 中的 X 数值排序
    """
    # 获取所有 .md 文件
    md_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.md')]

    # 定义排序 key 函数：提取 _page_ 后的数字
    def sort_key(file_path: str) -> int:
        filename = os.path.basename(file_path)
        match = re.search(r'_page_(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            # 如果没有找到数字，则排在最后
            return float('inf')

    # 按照数字排序
    sorted_files = sorted(md_files, key=sort_key)
    return sorted_files


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
    # file_path = "/home/user/documents/example.txt"
    # print(get_filename(file_path))          # 输出: example.txt
    # print(get_filename(file_path, False))  # 输出: example

    # 使用示例
    directory_to_delete = "E:\\my_project\\graph_rag_demo\\output\\demo_pdf1"
    delete_directory_if_non_empty(directory_to_delete)