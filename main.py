import gradio as gr
from utils.common_utils import get_sorted_md_files,delete_directory_if_non_empty,get_filename
from utils.log_utils import log
import os
from typing import List, Dict
from splitters.splitter_md import MarkdownDirSplitter
from milvus_db.milvus_db_with_schema import do_save_to_milvus


from dots_ocr.parser import do_parse
# 延迟导入 PaddleOCR，避免启动时的兼容性问题
# from paddle_ocr.paddle_parser import PaddleOCRParser

# md存储的临时模型output\
base_md_dir = r'F:\workspace\langgraph_project\Multimodal_RAG\output'
paddle_ocr_output_dir = r'F:\workspace\langgraph_project\Multimodal_RAG\paddle_ocr_output'

class ProcessorAPP:
    
    def __init__(self):
        # DotsOCR 相关属性
        self.pdf_path = None        # 当前上传的PDF路径
        self.md_dir = None          # 保存MD文件目录路径  假设 self.pdf_path = "F:\\docs\\example.pdf" 则 解析后的文件会放在 self.md_dir = 'F:\workspace\langgraph_project\Multimodal_RAG\output\example\'
        self.md_files = None        # 获取所有MD文件并按页码排序 self.md_files 是一个列表，里面存的是所有 .md 文件的完整路径 ["F:\\output\\page_1.md", "F:\\output\\page_2.md", "F:\\output\\page_3.md"]
        self.file_contents = {}     # self.file_contents 是 字典（dict），它的作用是：用文件的完整路径作为"钥匙（key）"，把对应的文件内容作为"值（value）"存起来，方便后续快速查找和显示
        
        # PaddleOCR-VL 相关属性
        self.paddle_pdf_path = None
        self.paddle_md_dir = None
        self.paddle_md_files = None
        self.paddle_file_contents = {}
        self.paddle_parser = None  # 延迟初始化
    
    def _init_paddle_parser(self):
        """延迟初始化 PaddleOCR 解析器"""
        if self.paddle_parser is None:
            from paddle_ocr.paddle_parser import PaddleOCRParser
            self.paddle_parser = PaddleOCRParser(output_base_dir=paddle_ocr_output_dir)

    def upload_pdf(self, pdf_file):
        log.info(f"上传pdf文件：{pdf_file}")
        self.pdf_path = pdf_file if pdf_file else None
        if self.pdf_path:
            return [
                f"PDF已上传: {os.path.basename(self.pdf_path)}",
                gr.update(interactive=True)   # 更新现有按钮为可交互
            ]
        else:
            return [
                "上传文件没有成功，请重新上传PDF文件",
                gr.update(interactive=False)  # 更新现有按钮为不可交互
            ]
    

    def parse_pdf(self):
        # 这解释了为什么前面要构造 md_files_dir = base_md_dir + filename —— 它正是 do_parse 实际写入的目录。
        """将用户上传的 PDF 文件解析成多个 Markdown(.md)文件，保存到指定目录中，并加载这些文件内容到内存缓存file_contents[f]，供后续界面选择和预览"""
        md_files_dir = os.path.join(base_md_dir, get_filename(self.pdf_path, False))  # eg: 'F:\workspace\langgraph_project\Multimodal_RAG\output\example\'
        delete_directory_if_non_empty(md_files_dir)   # 如果该目录已存在且非空，则删除整个目录及其内容。

        #  do_parse 会在传入的 output 目录下再次创建以PDF文件名命名的子目录 所以我们 output 选择上级目录
        do_parse(input_path=self.pdf_path, num_thread=32, no_fitz_preprocess=True, output=base_md_dir)   # 利用 dots_ocr 将 PDF 逐页解析为 Markdown 文件，输出到 base_md_dir 目录中，do_parse会自动创建以PDF文件名命名的子目录

        if os.path.isdir(md_files_dir):      # 检查 path 是否是一个存在的目录。
            self.md_dir = md_files_dir       # 保存当前 PDF 对应的 Markdown 输出目录，供后续使用
            log.info(f"🐶PDF已解析，生成了{len(os.listdir(md_files_dir))}个md文件")

            self.md_files = get_sorted_md_files(self.md_dir)    # 获取所有 .md 文件的路径列表（排序） ["F:\\output\\page_1.md", "F:\\output\\page_2.md", "F:\\output\\page_3.md"]
            log.info(f"🐶PDF已解析，生成的MD文件列表：{self.md_files}")
            # 把第一个文件的内容展示出来
            # 读取所有的md文件内容
            """最终处理之后的 self.file_contents:
            self.file_contents = {
                "F:\\output\\example\\page_1.md": "# 标题\n这是第一页内容...",
                "F:\\output\\example\\page_2.md": "## 第二页\n这里是表格..."
            }
            """
            # 每次循环，f 就是一个完整的文件路径（字符串）f="F:\\output\\example\\page_1.md",f="F:\\output\\example\\page_2.md"....
            for f in self.md_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:        # 打开文件 f（比如 page_1.md）file.read()：把整个文件内容读成一个大字符串
                        self.file_contents[f] = file.read() #   self.file_contents[f] = ...：把这个字符串存到字典里，用文件路径当“钥匙”（key）
                except Exception as e:
                    print(f"读取文件 {f} 时出错: {e}")
                    self.file_contents[f] = f"读取文件内容时出错: {e}"

            # 根据解析是否成功，动态更新 Gradio 界面上多个组件的状态。
            file_names = [os.path.basename(f) for f in self.md_files]

            return [
                f"🐶解析完成，共 {len(self.md_files)} 个MD文件",  # status
                gr.Dropdown(choices=file_names, label="MD文件列表", interactive=True),  # file_dropdown 注意下拉列表
                gr.update(interactive=False),  # parse_btn - 统一使用 gr.update
                gr.update(interactive=True)  # save_btn - 使用 gr.update
            ]

        else:
            return [
                "🐶解析失败！",  # status
                gr.Dropdown(interactive=False),  # file_dropdown
                gr.update(interactive=True),  # parse_btn - 统一使用 gr.update
                gr.update(interactive=False)  # save_btn - 使用 gr.update
            ]

    def select_md_file(self, selected_file):
        """Gradio 界面中“用户从下拉框选择一个 .md 文件”时触发的回调函数，目的是显示选中文件的内容"""
        log.info(f"🐶选择文件：{selected_file}")
        if selected_file:
            show_file = None
            # 根据显示的文件名（不含路径）找到完整路径 self.md_files 是之前 parse_pdf 时生成的完整路径列表 ["F:\\output\\example\\page_1.md", "F:\\output\\example\\page_2.md", "F:\\output\\example\\page_3.md"]
            for f in self.md_files:
                if os.path.basename(f) == selected_file:               # 比较“当前文件的纯文件名”是否等于“用户选中的文件名”。
                    show_file = f                                      # 如果相等 → 说明找到了！把完整路径 f 赋值给 show_file，并 break 跳出循环
                    break
            if show_file and show_file in self.file_contents:       # show_file 是通过用户选择的“短文件名”反查出来的“完整路径”，它正是 self.file_contents 字典中存储内容所用的 key。
                return self.file_contents[show_file]  # 从缓存读取 里面 key 的内容 返回给 content 文本框
            else:
                return "🐶没有找到该文件"
        else:
            return "🐶文件内容加载失败,选择的文件不对"

    def save_to_knowledge(self):
        """存入知识库"""
        if not self.md_dir:
            return "请先解析PDF文件"

        self.splitter = MarkdownDirSplitter(images_output_dir=r'F:\workspace\langgraph_project\Multimodal_RAG\output\images')
        result = self.splitter.process_md_dir(self.md_dir, self.pdf_path)
        res: List[Dict] = do_save_to_milvus(result)
        # 打印结果
        for i, doc in enumerate(res):
            print(f"\n文档 #{i + 1}:")
            print(doc['text'], doc['image_path'])
        return f"成功存入 {len(res)} 个文档到Milvus"
    
    # ==================== PaddleOCR-VL 相关方法 ====================
    
    def browse_existing_results(self, result_dir):
        """浏览已有的解析结果目录"""
        if not result_dir or not os.path.exists(result_dir):
            return [
                "❌ 目录不存在",
                gr.Dropdown(interactive=False),
                gr.update(interactive=False)
            ]
        
        try:
            # 延迟初始化解析器
            self._init_paddle_parser()
            
            # 获取该目录下的所有 MD 文件
            md_files = self.paddle_parser.get_markdown_files(result_dir)
            
            if not md_files:
                return [
                    "⚠️ 该目录下没有找到 MD 文件",
                    gr.Dropdown(interactive=False),
                    gr.update(interactive=False)
                ]
            
            # 保存状态
            self.paddle_md_dir = result_dir
            self.paddle_md_files = md_files
            
            # 读取所有 MD 文件内容
            self.paddle_file_contents = {}  # 清空之前的内容
            for f in self.paddle_md_files:
                try:
                    content = self.paddle_parser.read_markdown_content(f)
                    self.paddle_file_contents[f] = content
                except Exception as e:
                    log.error(f"读取文件 {f} 时出错: {e}")
                    self.paddle_file_contents[f] = f"读取文件内容时出错: {e}"
            
            # 生成文件名列表
            file_names = [os.path.basename(f) for f in self.paddle_md_files]
            
            return [
                f"✅ 加载成功！共找到 {len(md_files)} 个 MD 文件",
                gr.Dropdown(choices=file_names, label="MD文件列表", interactive=True),
                gr.update(interactive=True)  # 启用保存按钮
            ]
            
        except Exception as e:
            error_msg = f"加载失败: {str(e)}"
            log.error(f"[PaddleOCR] {error_msg}")
            import traceback
            traceback.print_exc()
            return [
                f"❌ {error_msg}",
                gr.Dropdown(interactive=False),
                gr.update(interactive=False)
            ]
    
    def upload_pdf_paddle(self, pdf_file):
        """PaddleOCR-VL: 上传PDF文件"""
        log.info(f"[PaddleOCR] 上传文件：{pdf_file}")
        self.paddle_pdf_path = pdf_file if pdf_file else None
        if self.paddle_pdf_path:
            return [
                f"✅ 文件已上传: {os.path.basename(self.paddle_pdf_path)}",
                gr.update(interactive=True)   # 启用解析按钮
            ]
        else:
            return [
                "❌ 上传失败，请重新上传文件",
                gr.update(interactive=False)  # 禁用解析按钮
            ]
    
    def parse_pdf_paddle(self):
        """PaddleOCR-VL: 解析PDF文件"""
        if not self.paddle_pdf_path:
            return [
                "❌ 请先上传文件",
                gr.Dropdown(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False)
            ]
        
        log.info(f"[PaddleOCR] 开始解析文件：{self.paddle_pdf_path}")
        
        try:
            # 延迟初始化 PaddleOCR 解析器
            self._init_paddle_parser()
            
            # 使用 PaddleOCR 解析文档
            success, message, output_dir = self.paddle_parser.parse_document(self.paddle_pdf_path)
            
            if success and output_dir:
                self.paddle_md_dir = output_dir
                
                # 获取生成的 Markdown 文件
                self.paddle_md_files = self.paddle_parser.get_markdown_files(output_dir)
                
                if not self.paddle_md_files:
                    return [
                        "⚠️ 解析完成，但未生成 Markdown 文件",
                        gr.Dropdown(interactive=False),
                        gr.update(interactive=True),
                        gr.update(interactive=False)
                    ]
                
                log.info(f"[PaddleOCR] 生成了 {len(self.paddle_md_files)} 个 MD 文件")
                
                # 读取所有 MD 文件内容
                for f in self.paddle_md_files:
                    try:
                        content = self.paddle_parser.read_markdown_content(f)
                        self.paddle_file_contents[f] = content
                    except Exception as e:
                        log.error(f"读取文件 {f} 时出错: {e}")
                        self.paddle_file_contents[f] = f"读取文件内容时出错: {e}"
                
                # 生成文件名列表（仅显示文件名，不含路径）
                file_names = [os.path.basename(f) for f in self.paddle_md_files]
                
                return [
                    f"✅ {message}",
                    gr.Dropdown(choices=file_names, label="MD文件列表", interactive=True),
                    gr.update(interactive=False),  # 禁用解析按钮
                    gr.update(interactive=True)    # 启用保存按钮
                ]
            else:
                return [
                    f"❌ {message}",
                    gr.Dropdown(interactive=False),
                    gr.update(interactive=True),
                    gr.update(interactive=False)
                ]
                
        except Exception as e:
            error_msg = f"解析过程出错: {str(e)}"
            log.error(f"[PaddleOCR] {error_msg}")
            import traceback
            traceback.print_exc()
            return [
                f"❌ {error_msg}",
                gr.Dropdown(interactive=False),
                gr.update(interactive=True),
                gr.update(interactive=False)
            ]
    
    def select_md_file_paddle(self, selected_file):
        """PaddleOCR-VL: 选择并显示 MD 文件内容"""
        log.info(f"[PaddleOCR] 选择文件：{selected_file}")
        if selected_file:
            show_file = None
            # 根据文件名找到完整路径
            for f in self.paddle_md_files:
                if os.path.basename(f) == selected_file:
                    show_file = f
                    break
            
            if show_file and show_file in self.paddle_file_contents:
                return self.paddle_file_contents[show_file]
            else:
                return "❌ 没有找到该文件"
        else:
            return "⚠️ 文件内容加载失败，选择的文件不对"
    
    def save_to_knowledge_paddle(self):
        """PaddleOCR-VL: 存入知识库"""
        if not self.paddle_md_dir:
            return "❌ 请先解析文件"
        
        try:
            splitter = MarkdownDirSplitter(images_output_dir=r'F:\workspace\langgraph_project\Multimodal_RAG\output\images')
            result = splitter.process_md_dir(self.paddle_md_dir, self.paddle_pdf_path)
            res: List[Dict] = do_save_to_milvus(result)
            
            # 打印结果
            for i, doc in enumerate(res):
                print(f"\n文档 #{i + 1}:")
                print(doc['text'], doc['image_path'])
            
            return f"✅ 成功存入 {len(res)} 个文档到 Milvus"
        except Exception as e:
            error_msg = f"存入知识库失败: {str(e)}"
            log.error(f"[PaddleOCR] {error_msg}")
            import traceback
            traceback.print_exc()
            return f"❌ {error_msg}"

    def create_interface(self):
        """创建一个构建多模态知识库的Gradio界面"""

        with gr.Blocks(title="多模态RAG - PDF解析与知识库构建") as app:
            gr.Markdown("# 🚀 多模态RAG - PDF解析与知识库构建系统")
            gr.Markdown("支持两种解析引擎：**DotsOCR** 和 **PaddleOCR-VL**")
            
            with gr.Tabs():
                # ==================== DotsOCR Tab ====================
                with gr.Tab("🔍 DotsOCR 解析器"):
                    gr.Markdown("### 使用 DotsOCR 进行文档解析")
                    
                    # 第一行：上传 + 解析按钮
                    with gr.Row():
                        pdf_upload = gr.File(label="📄 上传PDF文件")
                        parse_btn = gr.Button("🔍 解析PDF", variant="primary", interactive=False)
                    
                    # 状态显示
                    status = gr.Textbox(label="状态", value="等待操作...", interactive=False)
                    
                    # 第二行：文件列表 + 内容预览
                    with gr.Row():
                        # MD文件列表
                        file_dropdown = gr.Dropdown(choices=[], label="📄 选择MD文件", interactive=False)
                        # MD文件内容
                        content = gr.Textbox(label="📝 内容预览", lines=20, interactive=False, placeholder='请选择MD文件')
                    
                    # 保存按钮
                    save_btn = gr.Button("💾 存入知识库", variant="secondary", interactive=False)
                    
                    # 绑定事件
                    pdf_upload.change(
                        fn=self.upload_pdf,
                        inputs=pdf_upload,
                        outputs=[status, parse_btn]
                    )
                    
                    parse_btn.click(
                        fn=self.parse_pdf,
                        inputs=[],
                        outputs=[status, file_dropdown, parse_btn, save_btn]
                    )
                    
                    file_dropdown.change(
                        fn=self.select_md_file,
                        inputs=file_dropdown,
                        outputs=content
                    )
                    
                    save_btn.click(
                        fn=self.save_to_knowledge,
                        inputs=[],
                        outputs=status
                    )
                
                # ==================== PaddleOCR-VL Tab ====================
                with gr.Tab("🎯 PaddleOCR-VL 解析器"):
                    gr.Markdown("### 使用 PaddleOCR-VL 进行文档解析")
                    gr.Markdown("⚡ **特点**: 支持109种语言，擅长识别复杂元素（表格、公式、图表等）")
                    
                    # 第一行：上传 + 解析按钮
                    with gr.Row():
                        paddle_pdf_upload = gr.File(label="📄 上传PDF/图片文件")
                        paddle_parse_btn = gr.Button("🎯 解析文档", variant="primary", interactive=False)
                    
                    # 状态显示
                    paddle_status = gr.Textbox(label="状态", value="等待操作...", interactive=False)
                    
                    # 第二行：文件列表 + 内容预览
                    with gr.Row():
                        # MD文件列表
                        paddle_file_dropdown = gr.Dropdown(choices=[], label="📄 选择MD文件", interactive=False)
                        # MD文件内容
                        paddle_content = gr.Textbox(label="📝 内容预览", lines=20, interactive=False, placeholder='请选择MD文件')
                    
                    # 保存按钮
                    paddle_save_btn = gr.Button("💾 存入知识库", variant="secondary", interactive=False)
                    
                    # 绑定事件
                    paddle_pdf_upload.change(
                        fn=self.upload_pdf_paddle,
                        inputs=paddle_pdf_upload,
                        outputs=[paddle_status, paddle_parse_btn]
                    )
                    
                    paddle_parse_btn.click(
                        fn=self.parse_pdf_paddle,
                        inputs=[],
                        outputs=[paddle_status, paddle_file_dropdown, paddle_parse_btn, paddle_save_btn]
                    )
                    
                    paddle_file_dropdown.change(
                        fn=self.select_md_file_paddle,
                        inputs=paddle_file_dropdown,
                        outputs=paddle_content
                    )
                    
                    paddle_save_btn.click(
                        fn=self.save_to_knowledge_paddle,
                        inputs=[],
                        outputs=paddle_status
                    )
                
                # ==================== 浏览已有结果 Tab ====================
                with gr.Tab("📂 浏览已有结果"):
                    gr.Markdown("### 📂 浏览 PaddleOCR-VL 已有解析结果")
                    gr.Markdown("💡 **提示**: 无需重新解析，直接加载之前的解析结果")
                    
                    # 输入目录路径
                    with gr.Row():
                        with gr.Column(scale=4):
                            browse_dir_input = gr.Textbox(
                                label="解析结果目录路径",
                                placeholder=r"例如: F:\workspace\langgraph_project\Multimodal_RAG\paddle_ocr_output\大论文_...",
                                value="",
                                lines=1
                            )
                            gr.Markdown(f"**💾 默认输出目录**: `{paddle_ocr_output_dir}`")
                        with gr.Column(scale=1):
                            browse_load_btn = gr.Button("📂 加载结果", variant="primary", size="lg")
                    
                    # 状态显示
                    browse_status = gr.Textbox(label="状态", value="请输入目录路径并点击加载...", interactive=False)
                    
                    # 文件列表 + 内容预览
                    with gr.Row():
                        browse_file_dropdown = gr.Dropdown(choices=[], label="📄 选择MD文件", interactive=False)
                        browse_content = gr.Textbox(label="📝 内容预览", lines=20, interactive=False, placeholder='请先加载结果，然后选择MD文件')
                    
                    # 保存按钮
                    browse_save_btn = gr.Button("💾 存入知识库", variant="secondary", interactive=False)
                    
                    # 绑定事件
                    browse_load_btn.click(
                        fn=self.browse_existing_results,
                        inputs=browse_dir_input,
                        outputs=[browse_status, browse_file_dropdown, browse_save_btn]
                    )
                    
                    browse_file_dropdown.change(
                        fn=self.select_md_file_paddle,
                        inputs=browse_file_dropdown,
                        outputs=browse_content
                    )
                    
                    browse_save_btn.click(
                        fn=self.save_to_knowledge_paddle,
                        inputs=[],
                        outputs=browse_status
                    )
            
            # 底部信息
            gr.Markdown("---")
            gr.Markdown("💡 **使用提示**: 上传文件 → 点击解析 → 查看结果 → 存入知识库")

        return app

if __name__ == "__main__":
    app = ProcessorAPP()
    interface = app.create_interface()
    interface.launch()
