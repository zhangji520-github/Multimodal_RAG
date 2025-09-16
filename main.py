import gradio as gr
from utils.common_utils import get_sorted_md_files,delete_directory_if_non_empty,get_filename
from utils.log_utils import log
import os


from dots_ocr.parser import do_parse
# md存储的临时模型
base_md_dir = r'F:\workspace\langgraph_project\Multimodal_RAG\output'


class ProcessorAPP:
    
    def __init__(self):
        self.pdf_path = None        # 当前上传的PDF路径
        self.md_dir = None          # 保存MD文件目录路径  假设 self.pdf_path = "F:\\docs\\example.pdf" 则 解析后的文件会放在 self.md_dir = 'F:\workspace\langgraph_project\Multimodal_RAG\output\example\'
        self.md_files = None        # 获取所有MD文件并按页码排序 self.md_files 是一个列表，里面存的是所有 .md 文件的完整路径 ["F:\\output\\page_1.md", "F:\\output\\page_2.md", "F:\\output\\page_3.md"]
        self.file_contents = {}     # 缓存所有MD文件内容，避免重复读取

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
        """将用户上传的 PDF 文件解析成多个 Markdown(.md)文件，保存到指定目录中，并加载这些文件内容到内存缓存file_contents[f]，供后续界面选择和预览"""
        md_files_dir = os.path.join(base_md_dir, get_filename(self.pdf_path, False))  # eg: 'F:\workspace\langgraph_project\Multimodal_RAG\output\example\'
        delete_directory_if_non_empty(md_files_dir)
        #  do_parse 会在传入的 output 目录下再次创建以PDF文件名命名的子目录 所以我们 output 选择上级目录
        do_parse(input_path=self.pdf_path, num_thread=32, no_fitz_preprocess=True, output=base_md_dir)   # 将 PDF 逐页解析为 Markdown 文件，输出到 base_md_dir 目录中，do_parse会自动创建以PDF文件名命名的子目录
        if os.path.isdir(md_files_dir):      # 检查 path 是否是一个存在的目录。
            self.md_dir = md_files_dir       # 保存MD文件目录路径 记录当前 PDF 对应的输出目录
            log.info(f"🐶PDF已解析，生成了{len(os.listdir(md_files_dir))}个md文件")
            self.md_files = get_sorted_md_files(self.md_dir)
            log.info(f"🐶PDF已解析，生成的MD文件列表：{self.md_files}")
            # 把第一个文件的内容展示出来
            # 读取所有的md文件内容
            """
            self.file_contents = {
                "F:\\output\\example\\page_1.md": "# 标题\n这是第一页内容...",
                "F:\\output\\example\\page_2.md": "## 第二页\n这里是表格..."
            }
            """
            for f in self.md_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        self.file_contents[f] = file.read()
                except Exception as e:
                    print(f"读取文件 {f} 时出错: {e}")
                    self.file_contents[f] = f"读取文件内容时出错: {e}"
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

    def create_interface(self):
        """创建一个构建多模态知识库的Gradio界面"""

        with gr.Blocks() as app:
            gr.Markdown("## 🐶PDF解析与知识库存储和构建")

            # 第一行：上传 + 解析按钮
            with gr.Row():
                pdf_upload = gr.File(label="😘上传PDF")       # gr.File()：文件上传组件，用户可以选择 PDF 文件。
                parse_btn = gr.Button("🔍 解析PDF", variant="primary", interactive=False)  # 按钮初始为灰色不可点击，因为还没上传文件。
            # # 状态显示
            status = gr.Textbox(label="状态", value="等待操作...", interactive=False)  # 显示当前操作状态（如“PDF上传成功”），用户不能编辑（interactive=False）
            # 第二行：文件列表 + 内容预览
            with gr.Row():
                # MD文件列表
                file_dropdown = gr.Dropdown(choices=[], label="📄 选择MD文件", interactive=False)     # gr.Dropdown：下拉选择框，初始为空（choices=[]），不可交互。
                # MD文件中的内容 
                content = gr.Textbox(label="📝 内容预览", lines=20, interactive=False, placeholder='请选择MD文件')  # gr.Textbox(lines=20)：多行文本框，用于显示选中 MD 文件的内容。
            # 保存按钮
            save_btn = gr.Button("💾 存入知识库", variant="secondary", interactive=False)

            # 绑定按钮点击事件 （change事件：当用户选择或上传文件时触发）
            pdf_upload.change(
                fn=self.upload_pdf,
                inputs=pdf_upload,
                outputs=[status, parse_btn]       # 更新 status 文本框 和 parse_btn 按钮。
            )

            parse_btn.click(
                fn=self.parse_pdf,
                inputs=[],
                outputs=[status, file_dropdown, parse_btn, save_btn]      # 更新状态、下拉列表、控制解析按钮的可用性和控制保存按钮的可用性
            )

            file_dropdown.change(
                fn=self.select_md_file,
                inputs=file_dropdown,          # 下拉框选择 md 文档
                outputs=content                # 更新Textbox内容
            )

        return app

if __name__ == "__main__":
    app = ProcessorAPP()
    interface = app.create_interface()
    interface.launch()
