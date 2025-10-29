from paddleocr import PaddleOCRVL
from env_utils import PADDLE_OCR_URI
import os
from typing import Optional, Tuple, List, Dict

class PaddleOCRParser:
    """PaddleOCR-VL解析器类"""
    
    def __init__(self, output_base_dir="F:\\workspace\\langgraph_project\\Multimodal_RAG\\paddle_ocr_output"):
        """
        初始化PaddleOCR-VL解析器
        
        参数:
            output_base_dir: 输出基础目录
        """
        self.output_base_dir = output_base_dir
        self.pipeline = None
        
    def _init_pipeline(self):
        """延迟初始化pipeline，避免启动时就连接服务"""
        if self.pipeline is None:
            print(f"初始化PaddleOCR-VL服务，地址: http://{PADDLE_OCR_URI}/v1")
            self.pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=f"http://{PADDLE_OCR_URI}/v1"
            )
    
    def parse_document(self, file_path: str, output_dir: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        解析文档（支持图片和PDF）
        
        参数:
            file_path: 文件路径
            output_dir: 输出目录，如果为None则自动生成
            
        返回:
            (success, message, output_dir) - 成功标志、消息、输出目录
        """
        try:
            # 初始化pipeline
            self._init_pipeline()
            
            # 自动生成输出目录
            if output_dir is None:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                output_dir = os.path.join(self.output_base_dir, filename)
            
            # 创建输出目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建输出目录: {output_dir}")
            
            print(f"\n正在使用PaddleOCR-VL处理文件: {file_path}")
            print("="*50)
            
            # 执行OCR识别
            output = self.pipeline.predict(file_path)
            
            # 处理每个结果
            for idx, res in enumerate(output):
                print(f"\n[结果 {idx + 1}]")
                print("-"*50)
                
                # 打印结果到控制台
                res.print()
                
                # 保存为JSON格式
                # json_path = os.path.join(output_dir, f"result_{idx}")
                # res.save_to_json(save_path=json_path)
                # print(f"✓ JSON结果已保存到: {json_path}")
                
                # 保存为Markdown格式
                markdown_path = os.path.join(output_dir, f"result_{idx}")
                res.save_to_markdown(save_path=markdown_path)
                print(f"✓ Markdown结果已保存到: {markdown_path}")
            
            print("\n" + "="*50)
            print(f"✓ 处理完成！共识别 {len(output)} 个结果")
            print(f"✓ 所有结果已保存到目录: {output_dir}")
            print("="*50)
            
            return True, f"解析成功！共生成 {len(output)} 个结果", output_dir
            
        except Exception as e:
            error_msg = f"解析失败: {str(e)}"
            print(f"\n✗ {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg, None
    
    def get_markdown_files(self, output_dir: str) -> List[str]:
        """
        获取输出目录中的所有Markdown文件（递归查找子目录）
        
        参数:
            output_dir: 输出目录
            
        返回:
            Markdown文件路径列表
        """
        if not os.path.exists(output_dir):
            return []
        
        md_files = []
        # 遍历目录及子目录
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        
        return sorted(md_files)
    
    def read_markdown_content(self, md_file_path: str) -> str:
        """
        读取Markdown文件内容
        
        参数:
            md_file_path: Markdown文件路径
            
        返回:
            文件内容
        """
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"读取文件失败: {str(e)}"


def test_paddleocr_vl(image_path, output_dir="F:\\workspace\\langgraph_project\\Multimodal_RAG\\paddle_ocr_output"):
    """
    使用PaddleOCR-VL进行文档解析（兼容旧的测试函数）
    
    参数:
        image_path: 图片路径（支持本地路径或URL）
        output_dir: 输出目录，用于保存结果
    """
    parser = PaddleOCRParser(output_base_dir=os.path.dirname(output_dir))
    success, message, result_dir = parser.parse_document(image_path, output_dir)
    return success

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数指定文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默认测试文件（支持图片和PDF）
        file_path = r"F:\workspace\langgraph_project\Multimodal_RAG\大论文_多智能体系统主动容错控制及其在无人机中的应用.pdf"  # 可以改为 "QQ20251025-115637.png" 测试图片
    
    print("="*50)
    print("PaddleOCR-VL 测试程序")
    print("="*50)
    print(f"使用官方PaddleOCRVL类进行文档解析")
    print(f"支持格式: 图片(PNG/JPG/JPEG)、PDF文档")
    print("="*50 + "\n")
    
    # 执行测试
    result = test_paddleocr_vl(file_path, output_dir=r"F:\workspace\langgraph_project\Multimodal_RAG\paddle_ocr_output")
    
    if result:
        print("\n测试成功！✓")
    else:
        print("\n测试失败！✗")
