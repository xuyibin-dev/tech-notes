# RTX 3060 6GB 本地部署 Qwen2.5-VL-3B：从环境配置到 VQA 测试

## 1. 项目背景

最近在做一个具身视觉项目，我负责其中的 VLM 场景理解模块。

本项目使用：

- Qwen2.5-VL-3B-Instruct
- Python
- PyTorch
- Transformers
- ModelScope
- Gradio

## 2. 硬件环境

本项目实际运行环境如下：

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- 显存：6GB
- CPU：Intel i9
- 操作系统：Windows 11
- Python：3.11
- PyTorch：CUDA 12.8 环境

首先检查 PyTorch 是否能够正常调用 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
```

实际输出：

```text
True
NVIDIA GeForce RTX 3060 Laptop GPU
12.8
```

由于 Qwen2.5-VL-3B-Instruct 使用 FP16 加载时，模型整体显存需求超过 RTX 3060 Laptop 的 6GB 显存，因此无法完全加载到 GPU。

最终采用：

```python
torch_dtype=torch.float16
device_map="auto"
```

由 Transformers / Accelerate 自动将部分模型参数放置到 CPU，实现 GPU + CPU 混合推理。

这种方式虽然推理速度会比全部加载到 GPU 慢一些，但能够在 6GB 显存的消费级显卡上稳定运行模型。

---

## 3. 创建 Python 环境

使用 Conda 创建独立环境：

```bash
conda create -n qwen-vl python=3.11 -y
conda activate qwen-vl
```

安装 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安装完成后再次检查 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"
```

如果能够正常输出显卡名称并且：

```text
torch.cuda.is_available()
```

返回：

```text
True
```

说明 PyTorch GPU 环境配置成功。

---

## 4. 安装项目依赖

安装 Qwen2.5-VL 推理相关依赖：

```bash
pip install transformers accelerate qwen-vl-utils pillow opencv-python
```

安装 ModelScope：

```bash
pip install modelscope
```

安装 Gradio：

```bash
pip install gradio
```

项目主要使用的依赖包括：

```text
PyTorch
Transformers
Accelerate
Qwen-VL-Utils
ModelScope
Gradio
Pillow
OpenCV
```

我的环境中主要版本为：

```text
Python 3.11
PyTorch 2.11.0+cu128
Transformers 5.16.1
Gradio 6.26.0
qwen-vl-utils 0.0.14
Accelerate 1.14.0
```

---

## 5. 下载 Qwen2.5-VL-3B-Instruct

本项目使用的模型为：

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

使用 ModelScope 下载：

```bash
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir D:\qwen-vl-demo\models\Qwen2.5-VL-3B-Instruct
```

下载完成后，项目目录大致如下：

```text
D:\qwen-vl-demo
├── images
│   └── desktop.jpg
├── models
│   └── Qwen2.5-VL-3B-Instruct
├── results
│   └── result.txt
├── app.py
├── batch_test.py
├── test_image.py
└── vlm_api.py
```

其中：

```text
test_image.py
```

用于单张图片视觉问答测试。

```text
batch_test.py
```

用于批量执行 VQA 测试。

```text
app.py
```

用于启动 Gradio Web Demo。

```text
vlm_api.py
```

用于将 VLM 场景理解能力封装为结构化接口，方便后续系统联调。

---

## 6. 单张图片 VQA 测试

模型加载部分：

```python
import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

MODEL_PATH = r"D:\qwen-vl-demo\models\Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28
)
```

这里使用：

```python
torch_dtype=torch.float16
```

降低模型显存占用。

同时使用：

```python
device_map="auto"
```

让系统根据显存情况自动进行设备分配。

测试图片为一个桌面场景。

测试问题：

```text
请详细描述这张图片中的桌面，并列出你看到的主要物品。
```

模型输出能够识别出：

- 矿泉水瓶
- 白色杯子
- 钥匙
- 橙子
- 黑色手机
- 黑色耳机

说明模型已经能够完成基本的桌面场景理解与物体识别。

---

## 7. 设计 50 组 VQA 测试

单次测试只能证明模型能够正常运行，并不能说明模型在不同类型问题上的稳定性。

因此我针对桌面场景设计了 50 个 VQA 问题。

测试共分为 6 类：

### 7.1 物体识别

主要测试模型能否正确识别桌面中的物体。

例如：

```text
桌面上有哪些物体？
```

```text
请列出你最确定的5个物体。
```

### 7.2 数量判断

测试模型对物体数量的理解能力。

例如：

```text
桌面上有几个杯子？
```

```text
桌面上有几部手机？
```

### 7.3 颜色识别

测试模型对物体颜色属性的识别能力。

例如：

```text
杯子是什么颜色？
```

```text
手机是什么颜色？
```

### 7.4 空间关系

测试模型对不同物体位置关系的理解能力。

例如：

```text
钥匙和橙子哪个更靠近图片中央？
```

```text
杯子位于图片的大致什么位置？
```

### 7.5 场景理解

测试模型是否能够理解整个场景的语义。

例如：

```text
这个场景更像办公场景还是生活场景？
```

```text
这个桌面是否比较整洁？
```

### 7.6 抗幻觉测试

这类问题故意加入错误前提，用于测试模型是否会顺着错误问题进行回答。

例如：

```text
桌面上的两部手机分别是什么颜色？
```

但实际图片中只有一部手机。

还有：

```text
桌面上的笔记本电脑是什么颜色？
```

但实际场景中并不存在笔记本电脑。

通过这种方式可以测试模型是否能够拒绝错误前提，而不是产生幻觉。

---

## 8. 批量执行 VQA

在 `batch_test.py` 中将问题保存为列表，并逐题执行。

核心循环：

```python
for i, question in enumerate(questions, 1):
    print(
        f"正在处理第 {i}/{len(questions)} 题：{question}"
    )
```

程序依次完成：

```text
读取图片
↓
构造多模态 Prompt
↓
调用 Qwen2.5-VL
↓
生成回答
↓
保存结果
```

最终模型回答统一保存到：

```text
results/result.txt
```

这样可以避免每道题都手动运行一次，提高测试效率。

---

## 9. VQA 测试结果

完成 50 组测试后，我根据原始图片建立人工 Ground Truth，并对模型答案进行逐题判断。

评分方式比较直接：

```text
回答正确：1分
回答错误：0分
```

最终结果如下：

| 测试类别 | 正确率 |
| --- | ---: |
| 物体识别 | 90% |
| 数量判断 | 87.5% |
| 颜色识别 | 100% |
| 空间关系 | 83.3% |
| 场景理解 | 100% |
| 抗幻觉 | 100% |
| **总体准确率** | **92%** |

最终：

```text
46 / 50
```

即 50 道问题中有 46 道判断正确。

整体准确率：

```text
92%
```

其中颜色识别、场景理解以及抗幻觉测试表现较好。

相对而言，主要问题集中在：

```text
细粒度物体识别
数量判断
空间关系
```

---

## 10. 失败案例分析

除了统计准确率，我还对错误结果进行了分析。

### 10.1 橙子被识别成苹果

在一个问题中，我要求模型列出最确定的几个物体。

模型将：

```text
橙子
```

错误识别为：

```text
苹果
```

说明对于形状、颜色相近的小型物体，VLM 仍然可能出现细粒度识别错误。

---

### 10.2 小目标数量判断不稳定

在后续 Gradio 测试中，我提问：

```text
桌子上有几个钥匙？
```

模型回答：

```text
三把钥匙
```

但从原始图片中可以看到实际存在更多钥匙。

这说明 VLM 对多个重叠、尺寸较小的目标进行精确计数时，稳定性不足。

因此实际系统中不应该完全依靠 VLM 完成精确目标计数。

---

### 10.3 空间位置判断存在误差

部分测试要求模型判断：

```text
左侧
右侧
中央
左上
右下
```

等位置关系。

模型能够给出粗略空间描述，但部分答案与真实位置存在偏差。

例如：

```text
rough_position = "左上"
```

实际目标更接近图片左侧或左下区域。

这说明 VLM 的空间理解更适合作为：

```text
语义级粗略位置判断
```

而不是：

```text
精确坐标定位
```

---

## 11. 为什么还需要 YOLO 和 RGB-D

经过测试后，我对 VLM 在整个具身视觉系统中的定位也更加明确。

VLM 更适合负责：

```text
场景理解
目标是否存在
物体属性描述
语义推理
```

而 YOLO 更适合负责：

```text
目标检测
Bounding Box
目标类别
检测置信度
```

RGB-D 则更适合负责：

```text
目标距离
深度信息
空间方向
三维位置
```

因此后续系统架构可以设计为：

```text
用户语音
↓
ASR
↓
目标名称提取
↓
VLM 场景理解
↓
YOLO 目标检测
↓
RGB-D 距离 / 方位计算
↓
TTS 语音播报
```

例如用户询问：

```text
帮我找桌上的水杯
```

VLM 可以判断：

```json
{
  "target": "水杯",
  "exists": true,
  "description": "白色杯子",
  "rough_position": "左侧"
}
```

YOLO 后续可以提供目标检测框。

RGB-D 再根据检测结果计算目标距离和方向。

最终系统可以生成类似：

```text
水杯在你的左前方，大约0.8米。
```

这里的具体距离需要由实际 RGB-D 模块计算，VLM 本身不负责精确测距。

---

## 12. Gradio Web Demo

为了方便展示和测试模型，我使用 Gradio 编写了一个简单的 Web Demo。

基本功能包括：

```text
上传图片
↓
输入中文问题
↓
调用 Qwen2.5-VL
↓
显示模型回答
```

启动：

```bash
python app.py
```

服务启动后访问：

```text
http://127.0.0.1:7860
```

即可通过浏览器进行视觉问答。

Demo 支持：

- 图片上传
- 中文问题输入
- VLM 推理
- 模型回答显示

相比命令行测试，这种方式更加直观，也更适合作为项目展示。

---

## 13. VLM 接口封装

为了方便后续团队联调，我没有让其他模块直接依赖测试脚本，而是单独创建：

```text
vlm_api.py
```

并封装：

```python
def analyze_image(image_path, target):
    ...
```

调用方式：

```python
result = analyze_image(
    r"D:\qwen-vl-demo\images\desktop.jpg",
    "水杯"
)
```

接口返回结构化数据，例如：

```json
{
  "target": "水杯",
  "exists": true,
  "description": "白色杯子",
  "rough_position": "左侧"
}
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| target | 当前需要查找的目标 |
| exists | 图片中是否存在目标 |
| description | 目标外观描述 |
| rough_position | 目标的大致语义位置 |

如果目标不存在，则返回类似：

```json
{
  "target": "水杯",
  "exists": false,
  "description": "",
  "rough_position": "未知"
}
```

这样后续其他模块只需要调用统一接口，不需要了解 Qwen2.5-VL 内部推理过程。

---

## 14. 当前项目结构

目前我的 VLM 模块目录大致如下：

```text
qwen-vl-demo
├── images
│   └── desktop.jpg
│
├── models
│   └── Qwen2.5-VL-3B-Instruct
│
├── results
│   └── result.txt
│
├── app.py
├── batch_test.py
├── test_image.py
├── vlm_api.py
└── README.md
```

不同文件分别负责：

```text
test_image.py
→ 单张图片测试

batch_test.py
→ 50组 VQA 批量测试

app.py
→ Gradio Web Demo

vlm_api.py
→ 系统联调接口
```

相比最开始只有一个模型测试脚本，现在已经形成了一个比较完整的 VLM 场景理解模块。

---

## 15. 项目总结

通过这个项目，我完整经历了：

```text
环境搭建
↓
模型下载
↓
Qwen2.5-VL 本地部署
↓
GPU / CPU 混合推理
↓
桌面场景视觉问答
↓
50组 VQA 测试设计
↓
批量模型评测
↓
Ground Truth 人工评分
↓
失败案例分析
↓
Gradio Web Demo
↓
VLM API 接口封装
```

最终在：

```text
NVIDIA RTX 3060 Laptop 6GB
```

环境下，使用：

```text
FP16
device_map="auto"
CPU Offload
```

成功运行 Qwen2.5-VL-3B-Instruct。

在 50 组桌面场景 VQA 测试中：

```text
46 / 50
```

整体人工评测准确率达到：

```text
92%
```

其中：

```text
颜色识别：100%
场景理解：100%
抗幻觉测试：100%
```

通过失败案例也发现：

- VLM 对细粒度小目标识别仍可能出现错误
- 多个小物体的精确数量判断不够稳定
- 粗略空间关系可以理解，但不适合精确定位

因此在具身视觉系统中，我认为更合理的方案是：

```text
VLM
负责语义理解

YOLO
负责精确目标检测

RGB-D
负责距离与空间定位
```

目前我的 VLM 场景理解模块已经完成模型部署、VQA 测试、评测分析、Web Demo 和结构化接口封装。

下一阶段将继续进行 VLM 与目标检测、RGB-D 等模块的系统联调。
