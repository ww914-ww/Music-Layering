# Music Layering · 人声伴奏分离工具

基于 **PyQt6 + Demucs** 的桌面人声/伴奏分离工具。支持多轨分离（人声、鼓、贝斯、其他，以及含吉他/钢琴的 6 轨模式）、逐轨试听与波形可视化、增益/降噪后处理，以及 WAV / MP3 导出。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyQt6](https://img.shields.io/badge/PyQt6-6.5%2B-green) ![Demucs](https://img.shields.io/badge/Demucs-4.0%2B-orange)

## 功能特性

- **拖拽导入**：支持 MP3 / WAV / FLAC / OGG / M4A / AAC / WMA，以及 MP4 / AVI / MOV / MKV / WEBM / FLV / WMV（视频自动调用 ffmpeg 提取音轨）
- **多轨分离**：4 轨（人声 / 鼓 / 贝斯 / 其他）或 6 轨（增加吉他 / 钢琴）
- **模型可选**：`htdemucs`（标准）、`htdemucs_ft`（精细，TTA 偏移）、`htdemucs_6s`（6轨）
- **参数可调**：TTA 随机偏移次数、片段重叠比例、分块处理、计算设备（自动 / CUDA / CPU）
- **双波形面板**：左侧原始音频，右侧分离结果多轨叠加；支持点击波形跳转
- **逐轨控制**：单轨播放/暂停/停止、独立音量增益（-24 ~ +12 dB）、降噪开关
- **自动回退**：GPU 显存不足时自动切换 CPU 继续分离
- **导出格式**：WAV（直接复制）与 MP3（320kbps，需 ffmpeg）

## 环境要求

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html)（视频提取与 MP3 导出需要，已在 PATH 中即可）
- 依赖见 `requirements.txt`，其中 `pydub`、`noisereduce` 为可选

## 安装与运行

```bash
pip install -r requirements.txt
python vocal_separator.py
```

Windows 下可直接双击 `启动应用.bat` 启动（需确保依赖安装在 bat 指向的 Python 解释器中）。

> 首次运行会自动下载 Demucs 模型权重（htdemucs 约 80MB）到 `~/.cache/torch/hub/`。

## 自动化测试

项目附带离屏自动化测试与端到端真实分离测试：

```bash
python qa_test.py   # 18 项功能测试（加载 / 边界 / 后处理 / 导出 / 播放器 / 清理）
python qa_e2e.py    # 端到端：用 htdemucs 真实分离一段音频并校验输出
```

测试细节与质量评估见 [`质量评估报告.md`](质量评估报告.md)。

## 项目结构

```
vocal_separator.py     # 主程序（播放器 / 波形组件 / 分离线程 / 主窗口）
requirements.txt       # 依赖清单
启动应用.bat# Windows 启动脚本
qa_test.py             # 离屏功能测试
qa_e2e.py              # 端到端分离测试
质量评估报告.md         # 代码质量与功能验证报告
answer/                # 分离结果输出目录（已 gitignore）
```

## 已知问题

详见质量评估报告。三个高优先级问题：

1. 视频文件加载后，提取出的临时音频会被提前清理，导致「视频 → 分离」流程失败
2. 静音音频输入会在归一化时除零，产出 NaN 坏文件
3. 播放中主播放按钮显示「暂停」，实际行为是从头重播

## 说明

本工具仅供个人学习与创作参考使用，请遵守版权与相关法律法规。