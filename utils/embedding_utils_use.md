# Embeddings Utils 使用文档

## 概述

embeddings_utils.py 是一个多模态嵌入向量生成工具模块，基于阿里云达摩院的多模态嵌入API，支持文本和图像的向量化处理，具备完善的限流、重试和错误处理机制。

## 配置参数

### 核心配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_MODEL` | "multimodal-embedding-v1" | 达摩院多模态嵌入模型名称 |
| `ALIBABA_API_KEY` | 环境变量 `DASHSCOPE_API_KEY` | 阿里云API密钥 |
| `RPM_LIMIT` | 120 | 每分钟最大请求次数 |
| `WINDOW_SECONDS` | 60 | 限流时间窗口（秒） |

### 重试和限制配置

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `RETRY_ON_429` | True | 是否在429状态码时重试 |
| `MAX_429_RETRIES` | 5 | 429状态码最大重试次数 |
| `BASE_BACKOFF` | 2.0 | 指数退避基础等待时间（秒） |
| `MAX_IMAGE_BYTES` | 3MB | 图片最大体积限制 |

## 主要API函数

### 1. FixedWindowRateLimiter 类

**用途**: 固定窗口速率限制器，控制API调用频率

```python
# 初始化限制器
limiter = FixedWindowRateLimiter(limit=120, window_seconds=60)

# 获取请求许可（会自动阻塞直到可以继续）
limiter.acquire()
```

**参数**:
- `limit`: 时间窗口内允许的最大请求数
- `window_seconds`: 时间窗口长度（秒）

### 2. image_to_base64()

**用途**: 将本地图片文件转换为base64编码

```python
api_image, store_image = image_to_base64("path/to/image.jpg")
```

**参数**:
- `img`: 本地图片文件路径

**返回值**:
- `Tuple[str, str]`: (base64编码的图像数据, 原始路径)

### 3. normalize_image()

**用途**: 规范化图像输入，支持URL和本地文件两种类型

```python
api_image, store_image = normalize_image("https://example.com/image.jpg")
api_image, store_image = normalize_image("local/path/image.jpg")
```

**参数**:
- `img`: 图像路径或URL字符串

**返回值**:
- `Tuple[str, str]`: (用于API的图像数据, 用于存储的图像标识)
- 如果图片无效或超过限制，返回 `("", "")`

**特性**:
- 自动检测URL和本地文件
- URL图片会进行HEAD请求检查可达性和大小
- 自动过滤超过3MB的图片

### 4. call_dashscope_once()

**用途**: 调用达摩院多模态嵌入API一次

```python
# 文本嵌入
input_data = [{'text': '这是一段文本'}]
success, embedding, status, retry_after = call_dashscope_once(input_data)

# 图像嵌入
input_data = [{'image': 'data:image/jpeg;base64,...'}]
success, embedding, status, retry_after = call_dashscope_once(input_data)
```

**参数**:
- `input_data`: 输入数据列表，格式为 `[{'text': '..'}]` 或 `[{'image': '..'}]`

**返回值**:
- `Tuple[bool, List[float], Optional[int], Optional[float]]`: (成功标志, 嵌入向量, HTTP状态码, 重试等待时间)

### 5. process_item_with_guard()

**用途**: 处理单个数据项，生成嵌入向量

```python
# 处理文本项
item = {'text': '这是文本内容'}
result = process_item_with_guard(item, mode='text')

# 处理图像项
item = {'image_path': 'path/to/image.jpg'}
api_image = 'data:image/jpeg;base64,...'
result = process_item_with_guard(item, mode='image', api_image=api_image)
```

**参数**:
- `item`: 原始数据项字典
- `mode`: 处理模式，'text' 或 'image'
- `api_image`: 当mode为'image'时使用的图像数据

**返回值**:
- `Dict`: 处理后的数据项，包含 `dense` 字段（嵌入向量）

### 6. build_work_items()

**用途**: 构建工作项列表，将数据拆分为文本项和图片项

```python
expanded_data = [
    {'text': '文本内容', 'image_path': 'path/to/image.jpg'},
    {'text': '另一段文本'}
]
work_items = build_work_items(expanded_data)
```

**参数**:
- `expanded_data`: 扩展后的数据列表

**返回值**:
- `List[Tuple[Dict, str, str]]`: 工作项列表，每个元素为(数据项 代表原始数据项的副本（比如包含 text、image_path、id 等字段）, 模式 只能是 'text' 或 'image', API图像数据)

## 使用示例

### 基本文本嵌入

```python
from utils.embeddings_utils import process_item_with_guard

# 准备文本数据
text_item = {'text': '这是需要向量化的文本内容'}

# 生成嵌入向量
result = process_item_with_guard(text_item, mode='text')
print(f"嵌入向量维度: {len(result['dense'])}")
```

### 图像嵌入

```python
from utils.embeddings_utils import normalize_image, process_item_with_guard

# 准备图像数据
image_item = {'image_path': 'path/to/image.jpg'}

# 规范化图像
api_image, store_image = normalize_image(image_item['image_path'])

if api_image:
    # 生成图像嵌入向量
    result = process_item_with_guard(image_item, mode='image', api_image=api_image)
    print(f"图像嵌入向量维度: {len(result['dense'])}")
```

### 批量处理

```python
from utils.embeddings_utils import build_work_items, process_item_with_guard

# 准备混合数据
data = [
    {'text': '文本1', 'image_path': 'image1.jpg'},
    {'text': '文本2'},
    {'image_path': 'image2.jpg'}
]

# 构建工作项
work_items = build_work_items(data)

# 批量处理
results = []
for item, mode, api_image in work_items:
    result = process_item_with_guard(item, mode, api_image)
    results.append(result)
```

## 注意事项

1. **环境变量**: 确保设置了 `DASHSCOPE_API_KEY` 环境变量
2. **限流机制**: 自动限制每分钟120次请求，超出会自动等待
3. **图片限制**: 自动过滤超过3MB的图片文件
4. **错误处理**: 所有函数都有完善的异常处理，失败时返回空向量
5. **依赖包**: 需要安装 `dashscope`, `requests` 等依赖包

## 数据格式

### 输入数据格式

```python
{
    'text': '文本内容',           # 可选，文本字段
    'image_path': '图片路径'      # 可选，图片路径或URL
}
```

### 输出数据格式

```python
{
    'text': '文本内容或"图片"',    # 文本内容或图片标识
    'image_path': '图片路径',     # 图片存储路径（如果有）
    'dense': [0.1, 0.2, ...]     # 嵌入向量，失败时为空列表
}
```
