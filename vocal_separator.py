#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人声伴奏分离桌面应用 v2.0
=========================
基于 PyQt6 + Demucs (htdemucs) 的音频人声与伴奏分离工具。

功能特性:
  - 拖拽或点击导入音频/视频文件 (MP3 / WAV / FLAC / MP4 等)
  - 多轨分离: 人声、鼓、贝斯、其他 (4轨) 或 6轨 (含吉他、钢琴)
  - 模型选择: htdemucs / htdemucs_ft / 6轨实验模型
  - 分离参数调节: TTA 随机偏移、片段重叠、分块处理、设备选择
  - 原始音频与分离结果的波形可视化 (多轨叠加)
  - 逐轨播放控制、暂停、停止、波形点击跳转
  - 后处理: 逐轨音量增益、降噪
  - 支持导出 WAV 和 MP3 格式
  - 视频文件支持: 自动提取音频后分离
  - GPU 内存不足时自动回退到 CPU 模式

依赖安装:
  pip install PyQt6 demucs soundfile sounddevice numpy pydub noisereduce

运行方式:
  python vocal_separator.py
"""

import sys
import os
import time
import tempfile
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
import torch

# ── PyQt6 导入 ──────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QFrame, QSizePolicy, QSplitter, QSlider, QCheckBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QScrollArea,
)
from PyQt6.QtCore import (
    Qt, QObject, QThread, pyqtSignal, QTimer, QRectF,
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QDragEnterEvent,
    QDropEvent, QPainterPath, QMouseEvent,
)

# ── Demucs 4.x 导入 ─────────────────────────────────────────────────────
try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import ModelLoadingError
    DEMUCS_AVAILABLE = True
except ImportError as e:
    print(f"错误: 请先安装 demucs → pip install demucs\n详情: {e}")
    sys.exit(1)

# ── 可选: 降噪库 ────────────────────────────────────────────────────────
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          主题与配色常量                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

COLOR_PRIMARY = "#4A90D9"
COLOR_VOCALS = "#27AE60"
COLOR_ACCOMPANIMENT = "#E67E22"
COLOR_ORIGINAL = "#7F8C8D"
COLOR_BACKGROUND = "#F5F6FA"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT = "#2C3E50"
COLOR_TEXT_SECONDARY = "#95A5A6"
COLOR_BORDER = "#E0E3E8"
COLOR_PLAYHEAD = "#E74C3C"
COLOR_DROP_BORDER = "#B0C4DE"

# Stem 中文名和颜色映射
STEM_LABELS = {
    "vocals": "人声", "drums": "鼓", "bass": "贝斯",
    "other": "其他", "guitar": "吉他", "piano": "钢琴",
}
STEM_COLORS = {
    "vocals": "#27AE60", "drums": "#E67E22", "bass": "#3498DB",
    "other": "#9B59B6", "guitar": "#1ABC9C", "piano": "#F39C12",
}

# 可用模型列表
MODEL_CHOICES = ["htdemucs", "htdemucs_ft", "5c90dfd2"]

# 支持的音频和视频扩展名
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
ALL_MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

STYLE_SHEET = """
QMainWindow { background-color: #F5F6FA; }
QGroupBox {
    font-size: 13px; font-weight: bold; color: #2C3E50;
    border: 1px solid #E0E3E8; border-radius: 8px;
    margin-top: 10px; padding-top: 18px; background-color: #FFFFFF;
}
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; }
QPushButton {
    border: none; border-radius: 5px; padding: 6px 16px;
    font-size: 12px; font-weight: bold; color: #FFFFFF; background-color: #4A90D9;
}
QPushButton:hover { background-color: #357ABD; }
QPushButton:pressed { background-color: #2C6AA0; }
QPushButton:disabled { background-color: #BDC3C7; color: #ECF0F1; }
QPushButton#btnStop { background-color: #E74C3C; }
QPushButton#btnStop:hover { background-color: #C0392B; }
QProgressBar {
    border: none; border-radius: 5px; background-color: #E0E3E8;
    text-align: center; font-size: 11px; color: #2C3E50; height: 22px;
}
QProgressBar::chunk { background-color: #4A90D9; border-radius: 5px; }
QLabel { color: #2C3E50; }
QFrame#dropZone {
    border: 2px dashed #B0C4DE; border-radius: 12px; background-color: #FAFBFC;
}
QFrame#dropZone:hover { border-color: #4A90D9; background-color: #EEF2F7; }
QSlider::groove:horizontal {
    border: none; height: 6px; background: #E0E3E8; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4A90D9; width: 14px; height: 14px;
    margin: -4px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #4A90D9; border-radius: 3px; }
QComboBox, QSpinBox, QDoubleSpinBox {
    border: 1px solid #E0E3E8; border-radius: 4px; padding: 4px 8px;
    background: #FFFFFF; font-size: 12px;
}
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         音频播放器类                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AudioPlayer(QObject):
    """基于 sounddevice 的音频播放器。支持播放 / 暂停 / 停止 / Seek。"""

    position_changed = pyqtSignal(float)
    state_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._sr = 44100
        self._start_sample = 0
        self._start_time = 0.0
        self._paused_position = 0.0
        self._state = "stopped"
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_position)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state == "playing"

    @property
    def duration(self) -> float:
        if self._data is None or self._sr == 0:
            return 0.0
        return len(self._data) / self._sr

    def current_position(self) -> float:
        if self._state == "stopped":
            return 0.0
        elif self._state == "paused":
            return self._paused_position
        else:
            return self._paused_position + (time.time() - self._start_time)

    def load(self, filepath: str):
        self.stop()
        try:
            self._data, self._sr = sf.read(filepath, always_2d=True)
            if self._data.ndim > 1 and self._data.shape[1] > 1:
                self._data = self._data.mean(axis=1)
            else:
                self._data = self._data.flatten()
        except Exception as e:
            raise RuntimeError(f"无法加载音频文件: {e}")

    def play(self, start_position: float = 0.0):
        if self._data is None or len(self._data) == 0:
            return
        sd.stop()
        self._start_sample = int(start_position * self._sr)
        if self._start_sample >= len(self._data):
            self._start_sample = 0
        self._paused_position = start_position
        try:
            sd.play(self._data[self._start_sample:], self._sr)
            self._start_time = time.time()
            self._state = "playing"
            self._timer.start()
            self.state_changed.emit("playing")
        except sd.PortAudioError as e:
            raise RuntimeError(f"音频设备错误: {e}")

    def pause(self):
        if self._state != "playing":
            return
        sd.stop()
        self._paused_position = self.current_position()
        self._state = "paused"
        self._timer.stop()
        self.state_changed.emit("paused")

    def set_position(self, seconds: float):
        seconds = max(0.0, min(seconds, self.duration))
        if self._state == "playing":
            sd.stop()
            self._start_sample = int(seconds * self._sr)
            self._paused_position = seconds
            try:
                sd.play(self._data[self._start_sample:], self._sr)
                self._start_time = time.time()
                self._timer.start()
            except sd.PortAudioError:
                self._state = "stopped"
                self._timer.stop()
                self.state_changed.emit("stopped")
        else:
            self._paused_position = seconds
            self._start_sample = int(seconds * self._sr)

    def stop(self):
        sd.stop()
        self._start_sample = 0
        self._paused_position = 0.0
        self._state = "stopped"
        self._timer.stop()
        self.state_changed.emit("stopped")
        self.position_changed.emit(0.0)

    def _update_position(self):
        pos = self.current_position()
        self.position_changed.emit(pos)
        if pos >= self.duration:
            self.stop()
            self.state_changed.emit("finished")
            self.position_changed.emit(0.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         波形显示组件                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class WaveformWidget(QWidget):
    """自定义波形绘制组件。支持多轨叠加、播放位置指示线、点击跳转。"""

    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tracks: list[dict] = []
        self._duration = 0.0
        self._play_position = 0.0
        self._sr = 44100
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_audio_data(self, filepath: str, track_label: str = "audio",
                       color: str = COLOR_ORIGINAL):
        try:
            data, sr = sf.read(filepath, always_2d=True)
            if data.ndim > 1 and data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data.flatten()
            self._sr = sr
            self._duration = len(data) / sr
            self._tracks = [{"label": track_label, "data": data, "color": color}]
            self._play_position = 0.0
            self.update()
        except Exception as e:
            print(f"波形加载失败: {e}")
            self.clear()

    def add_track(self, filepath: str, track_label: str, color: str):
        try:
            data, sr = sf.read(filepath, always_2d=True)
            if data.ndim > 1 and data.shape[1] > 1:
                data = data.mean(axis=1)
            else:
                data = data.flatten()
            self._tracks.append({"label": track_label, "data": data, "color": color})
            self._duration = max(self._duration, len(data) / sr)
            self.update()
        except Exception as e:
            print(f"无法添加轨道 {track_label}: {e}")

    def set_play_position(self, seconds: float):
        self._play_position = max(0.0, min(seconds, self._duration))
        self.update()

    def clear(self):
        self._tracks = []
        self._duration = 0.0
        self._play_position = 0.0
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self._duration > 0 and event.button() == Qt.MouseButton.LeftButton:
            ratio = event.position().x() / self.width()
            self.seek_requested.emit(ratio * self._duration)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        painter.fillRect(0, 0, w, h, QColor(COLOR_CARD))
        if not self._tracks:
            painter.setPen(QColor(COLOR_TEXT_SECONDARY))
            painter.setFont(QFont("Microsoft YaHei", 11))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "暂无音频数据")
            painter.end()
            return
        num = len(self._tracks)
        th = (h - 10) / num
        for idx, track in enumerate(self._tracks):
            self._draw_track(painter, track["data"], track["color"],
                             track["label"], int(idx * th), int(th) - 4, w)
        if self._play_position > 0 and self._duration > 0:
            x = int((self._play_position / self._duration) * w)
            painter.setPen(QPen(QColor(COLOR_PLAYHEAD), 2))
            painter.drawLine(x, 0, x, h)
        painter.end()

    def _draw_track(self, painter, data, color_hex, label, top, height, width):
        if len(data) == 0 or height <= 0:
            return
        mid_y = top + height // 2
        half_h = height // 2 - 2
        target = width * 2
        if len(data) > target:
            spp = len(data) // target
            if spp > 1:
                trimmed = data[:spp * target]
                envelope = np.max(np.abs(trimmed.reshape(target, spp)), axis=1)
            else:
                envelope = np.abs(data[:target])
        else:
            envelope = np.abs(data)
        if len(envelope) == 0:
            return
        mx = np.max(envelope)
        if mx > 0:
            envelope = envelope / mx * half_h
        color = QColor(color_hex)
        path = QPainterPath()
        hp = len(envelope) // 2
        step = width / hp if hp >= 1 else 1
        path.moveTo(0, mid_y)
        for i in range(hp):
            path.lineTo(int(i * step), int(mid_y - envelope[i]))
        for i in range(hp - 1, -1, -1):
            path.lineTo(int(i * step), int(mid_y + envelope[i]))
        path.closeSubpath()
        fc = QColor(color)
        fc.setAlpha(60)
        painter.fillPath(path, QBrush(fc))
        painter.setPen(QPen(color, 1.0))
        painter.drawPath(path)
        painter.setPen(QColor(color))
        font = QFont("Microsoft YaHei", 8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(8, top + 14, label)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      音频分离工作线程                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AudioSeparationWorker(QThread):
    """后台线程: Demucs 分离，支持参数调节和多轨输出。"""

    progress_text = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished_separation = pyqtSignal(dict)  # {"stems": {name: path}, "sr": int}
    error_occurred = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str,
                 model_name: str = "htdemucs", shifts: int = 1,
                 overlap: float = 0.25, split: bool = True,
                 device: str = None, parent=None):
        super().__init__(parent)
        self._input_path = input_path
        self._output_dir = output_dir
        self._model_name = model_name
        self._shifts = shifts
        self._overlap = overlap
        self._split = split
        self._device_override = device

    def run(self):
        try:
            device = self._device_override or self._detect_device()
            self._run_separation(device)
        except torch.cuda.OutOfMemoryError:
            self.progress_text.emit("GPU 内存不足，回退到 CPU 模式...")
            try:
                self._run_separation("cpu")
            except Exception as e:
                self.error_occurred.emit(f"分离失败 (CPU回退也失败): {str(e)}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.progress_text.emit("内存不足，尝试回退到 CPU...")
                try:
                    self._run_separation("cpu")
                except Exception as e2:
                    self.error_occurred.emit(f"分离失败: {str(e2)}")
            else:
                self.error_occurred.emit(f"运行时错误: {str(e)}")
        except FileNotFoundError:
            self.error_occurred.emit(f"找不到输入文件: {self._input_path}")
        except ModelLoadingError as e:
            self.error_occurred.emit(
                f"模型加载失败: {e.args[0]}\n"
                f"可尝试删除缓存: {os.path.expanduser('~/.cache/torch/hub/')}"
            )
        except Exception as e:
            self.error_occurred.emit(f"分离异常:\n{str(e)}\n\n{traceback.format_exc()}")

    def _run_separation(self, device: str):
        self.progress_text.emit(f"正在加载模型 {self._model_name} (首次运行需下载)...")
        self.progress_value.emit(5)

        model = pretrained.get_model(self._model_name)
        model.to(device)
        model.eval()

        self.progress_text.emit(f"模型就绪 (设备: {device.upper()})，读取音频...")
        self.progress_value.emit(15)

        wav = self._load_audio(model)
        if wav is None:
            self.error_occurred.emit(
                f"无法读取音频: {self._input_path}\n"
                "请尝试转换为 WAV 格式或安装 ffmpeg。"
            )
            return

        ref = wav.mean(0)
        wav = wav - ref.mean()
        wav = wav / ref.std()

        self.progress_text.emit("正在进行分离 (可能需要 1~5 分钟)...")
        self.progress_value.emit(30)

        sources = apply_model(
            model, wav[None], device=device,
            shifts=self._shifts, split=self._split,
            overlap=self._overlap, progress=False,
        )[0]

        sources = sources * ref.std() + ref.mean()

        self.progress_text.emit("保存分离结果...")
        self.progress_value.emit(80)

        # 逐个保存所有 stem
        source_names = model.sources
        sr = model.samplerate
        stem_files = {}

        for i, name in enumerate(source_names):
            tensor = sources[i]  # (channels, samples)
            np_data = tensor.cpu().numpy().T  # (samples, channels)
            filepath = os.path.join(self._output_dir, f"{name}.wav")
            sf.write(filepath, np_data, sr)
            stem_files[name] = filepath

        self.progress_value.emit(100)
        self.progress_text.emit("分离完成!")
        self.finished_separation.emit({"stems": stem_files, "sr": sr})

    def _load_audio(self, model):
        import torchaudio as ta
        from demucs.audio import convert_audio
        try:
            return AudioFile(self._input_path).read(
                streams=0, samplerate=model.samplerate,
                channels=model.audio_channels,
            )
        except Exception:
            pass
        try:
            wav, sr = ta.load(self._input_path)
            return convert_audio(wav, sr, model.samplerate, model.audio_channels)
        except Exception:
            return None

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     Stem 控件行 (动态生成)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StemControlRow:
    """单个分离轨道的 UI 控件行，包含播放、音量、降噪、导出等功能。"""

    def __init__(self, stem_name: str, parent: QWidget):
        self.stem_name = stem_name
        self.label_cn = STEM_LABELS.get(stem_name, stem_name)
        self.color = STEM_COLORS.get(stem_name, "#95A5A6")

        self.container = QWidget(parent)
        self.container.setFixedHeight(42)
        layout = QHBoxLayout(self.container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # 颜色指示器
        color_dot = QLabel("●")
        color_dot.setStyleSheet(f"color: {self.color}; font-size: 16px; border: none; background: transparent;")
        color_dot.setFixedWidth(18)
        layout.addWidget(color_dot)

        # Stem 名称
        name_lbl = QLabel(self.label_cn)
        name_lbl.setFixedWidth(40)
        name_lbl.setStyleSheet("font-size: 12px; font-weight: bold; border: none; background: transparent;")
        layout.addWidget(name_lbl)

        # 播放按钮
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(32, 28)
        self.btn_play.setStyleSheet("font-size: 11px; padding: 2px;")
        layout.addWidget(self.btn_play)

        # 音量滑块
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(-24, 12)
        self.vol_slider.setValue(0)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.setToolTip("音量增益 (dB)")
        layout.addWidget(self.vol_slider)

        # 音量数值
        self.lbl_db = QLabel("0 dB")
        self.lbl_db.setFixedWidth(40)
        self.lbl_db.setStyleSheet("font-size: 11px; border: none; background: transparent;")
        self.vol_slider.valueChanged.connect(
            lambda v: self.lbl_db.setText(f"{v:+d} dB")
        )
        layout.addWidget(self.lbl_db)

        # 降噪开关
        self.chk_denoise = QCheckBox("降噪")
        self.chk_denoise.setStyleSheet("font-size: 11px; border: none; background: transparent;")
        if not HAS_NOISEREDUCE:
            self.chk_denoise.setEnabled(False)
            self.chk_denoise.setToolTip("需要安装: pip install noisereduce")
        layout.addWidget(self.chk_denoise)

        # 导出 WAV
        self.btn_export_wav = QPushButton("WAV")
        self.btn_export_wav.setFixedSize(44, 28)
        self.btn_export_wav.setStyleSheet("font-size: 10px; padding: 2px; background-color: #27AE60;")
        layout.addWidget(self.btn_export_wav)

        # 导出 MP3
        self.btn_export_mp3 = QPushButton("MP3")
        self.btn_export_mp3.setFixedSize(44, 28)
        self.btn_export_mp3.setStyleSheet("font-size: 10px; padding: 2px; background-color: #E67E22;")
        layout.addWidget(self.btn_export_mp3)

        layout.addStretch()

    def set_enabled(self, enabled: bool):
        self.btn_play.setEnabled(enabled)
        self.btn_export_wav.setEnabled(enabled)
        self.btn_export_mp3.setEnabled(enabled)

    def set_playing_highlight(self, active: bool):
        if active:
            self.btn_play.setStyleSheet(
                f"font-size: 11px; padding: 2px; background-color: {self.color}; color: #FFF;"
            )
        else:
            self.btn_play.setStyleSheet("font-size: 11px; padding: 2px;")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          主窗口应用                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    """主应用窗口。包含拖拽区、高级设置、双波形显示、逐轨控制、后处理、导出。"""

    def __init__(self):
        super().__init__()

        # ── 状态变量 ───────────────────────────────────────────────────
        self._input_file = None
        self._stem_files: dict[str, str] = {}          # stem名 → 文件路径
        self._processed_stem_files: dict[str, str] = {} # 后处理后的文件路径
        self._stem_rows: dict[str, StemControlRow] = {} # stem名 → UI控件行
        self._separated_current_track = None
        self._separation_done = False
        self._temp_dir = None
        self._video_extracted_wav = None  # 视频提取的临时音频

        # 播放器
        self._original_player = AudioPlayer(self)
        self._separated_player = AudioPlayer(self)

        # ── 初始化界面 ─────────────────────────────────────────────────
        self._init_ui()
        self._connect_signals()

        # ── 窗口设置 ───────────────────────────────────────────────────
        self.setWindowTitle("人声伴奏分离工具 v2.0")
        self.setMinimumSize(960, 720)
        self.resize(1060, 820)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.width()) // 2,
                   (screen.height() - self.height()) // 2)

    # ═══════════════════════════════════════════════════════════════════════
    #  界面初始化
    # ═══════════════════════════════════════════════════════════════════════

    def _init_ui(self):
        """构建完整界面布局，使用 QScrollArea 包裹以适配小屏幕。"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setCentralWidget(scroll)

        central = QWidget()
        scroll.setWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 14, 20, 14)

        # ── 标题 ──────────────────────────────────────────────────────
        title = QLabel("🎵 人声伴奏分离工具 v2.0")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2C3E50; padding: 4px 0;")
        main_layout.addWidget(title)

        # ── 拖拽区域 ──────────────────────────────────────────────────
        self._drop_zone = QFrame()
        self._drop_zone.setObjectName("dropZone")
        self._drop_zone.setMinimumHeight(90)
        self._drop_zone.setMaximumHeight(120)
        self._drop_zone.setAcceptDrops(True)
        drop_layout = QVBoxLayout(self._drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label = QLabel("拖拽音频/视频文件到此处\n或点击下方按钮选择文件")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label.setStyleSheet(
            "font-size: 14px; color: #7F8C8D; border: none; background: transparent; padding: 6px;"
        )
        drop_layout.addWidget(self._drop_label)
        self._btn_select = QPushButton("选择文件")
        self._btn_select.setFixedWidth(120)
        drop_layout.addWidget(self._btn_select, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self._drop_zone)

        # ── 文件信息栏 + 设备 ─────────────────────────────────────────
        info_layout = QHBoxLayout()
        self._lbl_file_info = QLabel("未选择文件")
        self._lbl_file_info.setStyleSheet(
            "font-size: 12px; color: #7F8C8D; padding: 3px 8px; background: #FAFBFC; border-radius: 4px;"
        )
        info_layout.addWidget(self._lbl_file_info)
        info_layout.addStretch()
        device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        self._lbl_device = QLabel(f"设备: {device_str}")
        self._lbl_device.setStyleSheet("font-size: 11px; color: #95A5A6;")
        info_layout.addWidget(self._lbl_device)
        main_layout.addLayout(info_layout)

        # ── 高级设置面板 (可折叠) ────────────────────────────────────
        self._advanced_group = QGroupBox("高级分离设置")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        self._advanced_group.setVisible(True)
        adv_layout = QHBoxLayout(self._advanced_group)
        adv_layout.setSpacing(12)

        adv_layout.addWidget(QLabel("模型:"))
        self._combo_model = QComboBox()
        self._combo_model.addItems(MODEL_CHOICES)
        self._combo_model.setCurrentText("htdemucs")
        adv_layout.addWidget(self._combo_model)

        adv_layout.addWidget(QLabel("TTA偏移:"))
        self._spin_shifts = QSpinBox()
        self._spin_shifts.setRange(1, 20)
        self._spin_shifts.setValue(1)
        self._spin_shifts.setToolTip("随机偏移次数，越大质量越高但越慢")
        adv_layout.addWidget(self._spin_shifts)

        adv_layout.addWidget(QLabel("重叠:"))
        self._spin_overlap = QDoubleSpinBox()
        self._spin_overlap.setRange(0.0, 1.0)
        self._spin_overlap.setSingleStep(0.05)
        self._spin_overlap.setValue(0.25)
        adv_layout.addWidget(self._spin_overlap)

        self._chk_split = QCheckBox("分块处理")
        self._chk_split.setChecked(True)
        adv_layout.addWidget(self._chk_split)

        adv_layout.addWidget(QLabel("设备:"))
        self._combo_device = QComboBox()
        self._combo_device.addItems(["自动", "CUDA", "CPU"])
        self._combo_device.setCurrentText("自动")
        adv_layout.addWidget(self._combo_device)

        adv_layout.addStretch()
        main_layout.addWidget(self._advanced_group)

        # ── 操作栏 ──────────────────────────────────────────────────
        action_layout = QHBoxLayout()
        self._btn_separate = QPushButton("开始分离")
        self._btn_separate.setFixedWidth(120)
        self._btn_separate.setEnabled(False)
        action_layout.addWidget(self._btn_separate)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("就绪")
        action_layout.addWidget(self._progress_bar)
        main_layout.addLayout(action_layout)

        # ── 双波形区域 ──────────────────────────────────────────────
        waveform_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧: 原始音频
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_group = QGroupBox("原始音频")
        left_gl = QVBoxLayout(left_group)
        self._original_waveform = WaveformWidget()
        left_gl.addWidget(self._original_waveform)
        orig_ctrl = QHBoxLayout()
        self._btn_play_orig = QPushButton("▶ 播放原曲")
        self._btn_play_orig.setEnabled(False)
        self._btn_pause_orig = QPushButton("⏸ 暂停")
        self._btn_pause_orig.setEnabled(False)
        self._btn_stop_orig = QPushButton("⏹ 停止")
        self._btn_stop_orig.setEnabled(False)
        self._btn_stop_orig.setObjectName("btnStop")
        orig_ctrl.addWidget(self._btn_play_orig)
        orig_ctrl.addWidget(self._btn_pause_orig)
        orig_ctrl.addWidget(self._btn_stop_orig)
        orig_ctrl.addStretch()
        self._lbl_orig_time = QLabel("00:00 / 00:00")
        self._lbl_orig_time.setStyleSheet("font-size: 11px; color: #7F8C8D;")
        orig_ctrl.addWidget(self._lbl_orig_time)
        left_gl.addLayout(orig_ctrl)
        left_layout.addWidget(left_group)

        # 右侧: 分离结果
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_group = QGroupBox("分离结果")
        right_gl = QVBoxLayout(right_group)
        right_gl.setSpacing(6)

        self._separated_waveform = WaveformWidget()
        right_gl.addWidget(self._separated_waveform)

        # Stem 控件行容器
        self._stem_rows_container = QVBoxLayout()
        self._stem_rows_container.setSpacing(2)
        right_gl.addLayout(self._stem_rows_container)

        # 全局控制 + 后处理按钮
        sep_ctrl = QHBoxLayout()
        self._btn_pause_sep = QPushButton("⏸ 暂停")
        self._btn_pause_sep.setEnabled(False)
        self._btn_stop_sep = QPushButton("⏹ 停止")
        self._btn_stop_sep.setEnabled(False)
        self._btn_stop_sep.setObjectName("btnStop")
        sep_ctrl.addWidget(self._btn_pause_sep)
        sep_ctrl.addWidget(self._btn_stop_sep)
        sep_ctrl.addStretch()
        self._lbl_sep_time = QLabel("00:00 / 00:00")
        self._lbl_sep_time.setStyleSheet("font-size: 11px; color: #7F8C8D;")
        sep_ctrl.addWidget(self._lbl_sep_time)
        right_gl.addLayout(sep_ctrl)

        # 后处理操作按钮
        pp_btn_layout = QHBoxLayout()
        self._btn_apply_postproc = QPushButton("应用后处理")
        self._btn_apply_postproc.setEnabled(False)
        self._btn_apply_postproc.setStyleSheet("background-color: #27AE60;")
        self._btn_reset_postproc = QPushButton("重置后处理")
        self._btn_reset_postproc.setEnabled(False)
        pp_btn_layout.addWidget(self._btn_apply_postproc)
        pp_btn_layout.addWidget(self._btn_reset_postproc)
        pp_btn_layout.addStretch()
        right_gl.addLayout(pp_btn_layout)

        right_layout.addWidget(right_group)
        waveform_splitter.addWidget(left_panel)
        waveform_splitter.addWidget(right_panel)
        waveform_splitter.setStretchFactor(0, 1)
        waveform_splitter.setStretchFactor(1, 2)
        main_layout.addWidget(waveform_splitter, stretch=1)

        # ── 状态栏 ───────────────────────────────────────────────────
        self._status_label = QLabel("就绪 - 请导入音频或视频文件")
        self._status_label.setStyleSheet("font-size: 11px; color: #95A5A6; padding: 2px 0;")
        main_layout.addWidget(self._status_label)

    def _connect_signals(self):
        """连接所有信号与槽"""
        self._btn_select.clicked.connect(self._on_select_file)
        self._btn_separate.clicked.connect(self._on_start_separation)

        # 原始音频
        self._btn_play_orig.clicked.connect(self._on_play_original)
        self._btn_pause_orig.clicked.connect(self._on_pause_original)
        self._btn_stop_orig.clicked.connect(self._on_stop_original)
        self._original_player.position_changed.connect(self._on_original_pos_changed)
        self._original_player.state_changed.connect(self._on_original_state_changed)
        self._original_waveform.seek_requested.connect(self._on_original_seek)

        # 分离结果
        self._btn_pause_sep.clicked.connect(self._on_pause_separated)
        self._btn_stop_sep.clicked.connect(self._on_stop_separated)
        self._separated_player.position_changed.connect(self._on_separated_pos_changed)
        self._separated_player.state_changed.connect(self._on_separated_state_changed)
        self._separated_waveform.seek_requested.connect(self._on_separated_seek)

        # 后处理
        self._btn_apply_postproc.clicked.connect(self._on_apply_postproc)
        self._btn_reset_postproc.clicked.connect(self._on_reset_postproc)

    # ═══════════════════════════════════════════════════════════════════════
    #  拖拽事件
    # ═══════════════════════════════════════════════════════════════════════

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                ext = Path(urls[0].toLocalFile()).suffix.lower()
                if ext in ALL_MEDIA_EXTS:
                    event.acceptProposedAction()
                    self._drop_zone.setStyleSheet(
                        "#dropZone { border: 2px dashed #4A90D9; border-radius: 12px; background-color: #EEF2F7; }"
                    )
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_zone.setStyleSheet(
            "#dropZone { border: 2px dashed #B0C4DE; border-radius: 12px; background-color: #FAFBFC; }"
            "#dropZone:hover { border-color: #4A90D9; background-color: #EEF2F7; }"
        )

    def dropEvent(self, event: QDropEvent):
        self.dragLeaveEvent(None)
        if event.mimeData().hasUrls():
            self._load_audio_file(event.mimeData().urls()[0].toLocalFile())

    # ═══════════════════════════════════════════════════════════════════════
    #  文件加载
    # ═══════════════════════════════════════════════════════════════════════

    def _on_select_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择音频/视频文件", "",
            "媒体文件 (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.mp4 *.avi *.mov *.mkv *.webm);;"
            "音频文件 (*.mp3 *.wav *.flac *.ogg *.m4a *.aac);;"
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.webm);;"
            "所有文件 (*.*)",
        )
        if filepath:
            self._load_audio_file(filepath)

    def _load_audio_file(self, filepath: str):
        """加载文件: 若为视频则先提取音频，然后验证、显示波形。"""
        try:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"文件不存在: {filepath}")

            ext = Path(filepath).suffix.lower()

            # 视频文件: 先提取音频
            if ext in VIDEO_EXTS:
                self._status_label.setText("正在从视频提取音频...")
                self._status_label.setStyleSheet("font-size: 11px; color: #E67E22;")
                QApplication.processEvents()
                filepath = self._extract_audio_from_video(filepath)
                if filepath is None:
                    return

            # 验证音频
            info = sf.info(filepath)
            self._input_file = filepath
            filename = os.path.basename(filepath)

            duration_str = self._format_duration(info.duration)
            self._lbl_file_info.setText(
                f"文件: {filename}  |  时长: {duration_str}  |  "
                f"采样率: {info.samplerate} Hz  |  声道: {info.channels}"
            )

            self._original_waveform.set_audio_data(filepath, "原始音频", COLOR_ORIGINAL)
            self._separated_waveform.clear()
            self._original_player.load(filepath)

            # 重置状态
            self._separation_done = False
            self._stem_files = {}
            self._processed_stem_files = {}
            self._separated_current_track = None
            self._clear_stem_rows()

            # 更新 UI
            self._btn_separate.setEnabled(True)
            self._btn_play_orig.setEnabled(True)
            self._btn_pause_orig.setEnabled(True)
            self._btn_stop_orig.setEnabled(True)
            self._btn_pause_sep.setEnabled(False)
            self._btn_stop_sep.setEnabled(False)
            self._btn_apply_postproc.setEnabled(False)
            self._btn_reset_postproc.setEnabled(False)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("就绪")
            self._status_label.setText(f"已加载: {filename}")
            self._status_label.setStyleSheet("font-size: 11px; color: #27AE60;")
            self._cleanup_temp_dir()

        except FileNotFoundError as e:
            QMessageBox.critical(self, "文件错误", str(e))
        except sf.LibsndfileError:
            QMessageBox.critical(self, "格式错误",
                f"无法读取文件，可能已损坏或格式不受支持:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载文件时出错:\n{str(e)}")

    def _extract_audio_from_video(self, video_path: str) -> str | None:
        """使用 ffmpeg 从视频中提取音频轨道，返回临时 WAV 路径。"""
        try:
            temp_wav = tempfile.mktemp(suffix=".wav")
            result = subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2", "-y", temp_wav,
            ], capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                QMessageBox.critical(self, "提取失败",
                    f"从视频提取音频失败:\n{result.stderr[-500:]}")
                return None
            self._video_extracted_wav = temp_wav
            self._status_label.setText("音频提取完成")
            return temp_wav
        except FileNotFoundError:
            QMessageBox.critical(self, "缺少 ffmpeg",
                "视频处理需要 ffmpeg，请先安装:\nhttps://ffmpeg.org/download.html")
            return None
        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, "提取超时", "视频音频提取超时，请尝试较短的视频。")
            return None

    # ═══════════════════════════════════════════════════════════════════════
    #  Stem 行管理
    # ═══════════════════════════════════════════════════════════════════════

    def _build_stem_rows(self, stem_names: list[str]):
        """根据 stem 名称列表动态生成控件行。"""
        self._clear_stem_rows()
        for name in stem_names:
            row = StemControlRow(name, None)
            row.set_enabled(True)
            # 连接播放
            row.btn_play.clicked.connect(
                lambda checked, n=name: self._on_play_separated(n)
            )
            # 连接导出
            row.btn_export_wav.clicked.connect(
                lambda checked, n=name: self._on_export(n, "wav")
            )
            row.btn_export_mp3.clicked.connect(
                lambda checked, n=name: self._on_export(n, "mp3")
            )
            self._stem_rows[name] = row
            self._stem_rows_container.addWidget(row.container)
        self._stem_rows_container.addStretch()

    def _clear_stem_rows(self):
        """清除所有动态生成的 stem 行。"""
        for row in self._stem_rows.values():
            row.container.setParent(None)
            row.container.deleteLater()
        self._stem_rows = {}
        # 清除 stretch items
        while self._stem_rows_container.count():
            item = self._stem_rows_container.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

    # ═══════════════════════════════════════════════════════════════════════
    #  分离操作
    # ═══════════════════════════════════════════════════════════════════════

    def _on_start_separation(self):
        if not self._input_file:
            return
        self._cleanup_temp_dir()
        self._temp_dir = tempfile.mkdtemp(prefix="vocal_sep_")

        # 禁用 UI
        self._btn_separate.setEnabled(False)
        self._btn_select.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("准备中...")
        self._status_label.setText("正在分离...")
        self._status_label.setStyleSheet("font-size: 11px; color: #E67E22;")

        # 读取高级设置
        model_name = self._combo_model.currentText()
        shifts = self._spin_shifts.value()
        overlap = self._spin_overlap.value()
        split = self._chk_split.isChecked()
        device_str = self._combo_device.currentText()
        device = None if device_str == "自动" else device_str.lower()

        self._worker = AudioSeparationWorker(
            self._input_file, self._temp_dir,
            model_name=model_name, shifts=shifts,
            overlap=overlap, split=split, device=device,
        )
        self._worker.progress_text.connect(self._on_separation_progress_text)
        self._worker.progress_value.connect(self._on_separation_progress_value)
        self._worker.finished_separation.connect(self._on_separation_finished)
        self._worker.error_occurred.connect(self._on_separation_error)
        self._worker.start()

    def _on_separation_progress_text(self, text: str):
        self._status_label.setText(text)
        self._status_label.setStyleSheet("font-size: 11px; color: #E67E22;")

    def _on_separation_progress_value(self, value: int):
        self._progress_bar.setValue(value)
        if value < 30:
            self._progress_bar.setFormat("准备中... %p%")
        elif value < 80:
            self._progress_bar.setFormat("分离中... %p%")
        else:
            self._progress_bar.setFormat("保存中... %p%")

    def _on_separation_finished(self, result: dict):
        stems = result["stems"]  # {"vocals": path, "drums": path, ...}
        sr = result["sr"]

        self._stem_files = stems
        self._processed_stem_files = {}  # 重置后处理
        self._separated_current_track = None
        self._separation_done = True

        # 生成 stem 控件行
        stem_names = list(stems.keys())
        self._build_stem_rows(stem_names)

        # 更新波形: 添加所有 stem 轨道
        self._separated_waveform.clear()
        for name in stem_names:
            label = STEM_LABELS.get(name, name)
            color = STEM_COLORS.get(name, "#95A5A6")
            self._separated_waveform.add_track(stems[name], label, color)

        # 启用控件
        self._btn_pause_sep.setEnabled(True)
        self._btn_stop_sep.setEnabled(True)
        self._btn_apply_postproc.setEnabled(True)
        self._btn_reset_postproc.setEnabled(True)
        self._btn_separate.setEnabled(True)
        self._btn_select.setEnabled(True)

        self._progress_bar.setValue(100)
        self._progress_bar.setFormat("完成!")
        self._status_label.setText(f"分离完成! 共 {len(stems)} 轨: " +
                                    ", ".join(STEM_LABELS.get(n, n) for n in stem_names))
        self._status_label.setStyleSheet("font-size: 11px; color: #27AE60;")

    def _on_separation_error(self, error_msg: str):
        self._btn_separate.setEnabled(True)
        self._btn_select.setEnabled(True)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("错误")
        self._status_label.setText("分离失败")
        self._status_label.setStyleSheet("font-size: 11px; color: #E74C3C;")
        QMessageBox.critical(self, "分离错误", error_msg)

    # ═══════════════════════════════════════════════════════════════════════
    #  原始音频播放控制
    # ═══════════════════════════════════════════════════════════════════════

    def _on_play_original(self):
        try:
            if self._original_player.state == "paused":
                self._original_player.play(self._original_player.current_position())
            else:
                self._original_player.play()
        except RuntimeError as e:
            QMessageBox.warning(self, "播放错误", str(e))

    def _on_pause_original(self):
        self._original_player.pause()

    def _on_stop_original(self):
        self._original_player.stop()

    def _on_original_pos_changed(self, position: float):
        self._original_waveform.set_play_position(position)
        self._lbl_orig_time.setText(
            f"{self._format_duration(position)} / {self._format_duration(self._original_player.duration)}"
        )

    def _on_original_state_changed(self, state: str):
        if state == "playing":
            self._btn_play_orig.setText("⏸ 暂停")
        elif state in ("paused", "stopped", "finished"):
            self._btn_play_orig.setText("▶ 播放原曲")
            if state == "finished":
                self._original_waveform.set_play_position(0)
                self._lbl_orig_time.setText(
                    f"00:00 / {self._format_duration(self._original_player.duration)}"
                )

    def _on_original_seek(self, seconds: float):
        self._original_player.set_position(seconds)
        self._original_waveform.set_play_position(seconds)
        self._lbl_orig_time.setText(
            f"{self._format_duration(seconds)} / {self._format_duration(self._original_player.duration)}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  分离结果播放控制
    # ═══════════════════════════════════════════════════════════════════════

    def _get_stem_filepath(self, stem_name: str) -> str | None:
        """获取 stem 的实际文件路径，优先使用后处理版本。"""
        if stem_name in self._processed_stem_files:
            return self._processed_stem_files[stem_name]
        return self._stem_files.get(stem_name)

    def _on_play_separated(self, stem_name: str):
        filepath = self._get_stem_filepath(stem_name)
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(self, "错误", f"{STEM_LABELS.get(stem_name, stem_name)} 文件不存在，请重新分离。")
            return

        try:
            if self._separated_current_track != stem_name:
                self._separated_player.load(filepath)
                self._separated_current_track = stem_name

            if self._separated_player.state == "paused":
                self._separated_player.play(self._separated_player.current_position())
            else:
                self._separated_player.play()

            # 更新高亮
            for name, row in self._stem_rows.items():
                row.set_playing_highlight(name == stem_name)
        except RuntimeError as e:
            QMessageBox.warning(self, "播放错误", str(e))

    def _on_pause_separated(self):
        self._separated_player.pause()

    def _on_stop_separated(self):
        self._separated_player.stop()
        self._separated_current_track = None
        for row in self._stem_rows.values():
            row.set_playing_highlight(False)

    def _on_separated_pos_changed(self, position: float):
        self._separated_waveform.set_play_position(position)
        self._lbl_sep_time.setText(
            f"{self._format_duration(position)} / {self._format_duration(self._separated_player.duration)}"
        )

    def _on_separated_state_changed(self, state: str):
        if state in ("stopped", "finished"):
            for row in self._stem_rows.values():
                row.set_playing_highlight(False)
            if state == "finished":
                self._separated_waveform.set_play_position(0)
                self._lbl_sep_time.setText(
                    f"00:00 / {self._format_duration(self._separated_player.duration)}"
                )

    def _on_separated_seek(self, seconds: float):
        self._separated_player.set_position(seconds)
        self._separated_waveform.set_play_position(seconds)
        self._lbl_sep_time.setText(
            f"{self._format_duration(seconds)} / {self._format_duration(self._separated_player.duration)}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  后处理
    # ═══════════════════════════════════════════════════════════════════════

    def _on_apply_postproc(self):
        """逐轨应用增益和降噪，生成新的临时文件并重新加载波形。"""
        if not self._stem_files:
            return

        self._status_label.setText("正在应用后处理...")
        self._status_label.setStyleSheet("font-size: 11px; color: #E67E22;")
        QApplication.processEvents()

        processed = {}
        for name, filepath in self._stem_files.items():
            row = self._stem_rows.get(name)
            if not row:
                processed[name] = filepath
                continue

            try:
                data, sr = sf.read(filepath, always_2d=True)

                # 增益
                gain_db = row.vol_slider.value()
                if gain_db != 0:
                    gain_linear = 10 ** (gain_db / 20.0)
                    data = data * gain_linear

                # 降噪
                if row.chk_denoise.isChecked() and HAS_NOISEREDUCE:
                    if data.ndim > 1:
                        for ch in range(data.shape[1]):
                            data[:, ch] = nr.reduce_noise(
                                y=data[:, ch], sr=sr, prop_decrease=0.85
                            )
                    else:
                        data = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.85)

                # 防止削波
                data = np.clip(data, -0.99, 0.99)

                # 保存到临时文件
                out_path = os.path.join(self._temp_dir, f"{name}_processed.wav")
                sf.write(out_path, data, sr)
                processed[name] = out_path
            except Exception as e:
                print(f"后处理 {name} 失败: {e}")
                processed[name] = filepath

        self._processed_stem_files = processed

        # 更新波形
        self._separated_waveform.clear()
        for name in self._stem_files:
            label = STEM_LABELS.get(name, name)
            color = STEM_COLORS.get(name, "#95A5A6")
            self._separated_waveform.add_track(
                self._get_stem_filepath(name), label, color
            )

        self._separated_current_track = None
        self._separated_player.stop()

        self._status_label.setText("后处理已应用")
        self._status_label.setStyleSheet("font-size: 11px; color: #27AE60;")

    def _on_reset_postproc(self):
        """重置后处理，恢复原始分离结果。"""
        self._processed_stem_files = {}

        # 重置滑块
        for row in self._stem_rows.values():
            row.vol_slider.setValue(0)
            row.chk_denoise.setChecked(False)

        # 更新波形
        self._separated_waveform.clear()
        for name, filepath in self._stem_files.items():
            label = STEM_LABELS.get(name, name)
            color = STEM_COLORS.get(name, "#95A5A6")
            self._separated_waveform.add_track(filepath, label, color)

        self._separated_current_track = None
        self._separated_player.stop()

        self._status_label.setText("后处理已重置")
        self._status_label.setStyleSheet("font-size: 11px; color: #27AE60;")

    # ═══════════════════════════════════════════════════════════════════════
    #  导出功能
    # ═══════════════════════════════════════════════════════════════════════

    def _on_export(self, stem_name: str, fmt: str):
        source_path = self._get_stem_filepath(stem_name)
        if not source_path or not os.path.exists(source_path):
            QMessageBox.warning(self, "错误", "源文件不存在，请重新分离。")
            return

        label = STEM_LABELS.get(stem_name, stem_name)
        ext_filter = f"{fmt.upper()} 文件 (*.{fmt})"
        default_name = f"{label}.{fmt}"

        save_path, _ = QFileDialog.getSaveFileName(
            self, f"导出{label} ({fmt.upper()})", default_name, ext_filter
        )
        if not save_path:
            return

        try:
            if fmt == "wav":
                shutil.copy2(source_path, save_path)
            elif fmt == "mp3":
                self._export_as_mp3(source_path, save_path, label)
            self._status_label.setText(f"{label}已导出: {os.path.basename(save_path)}")
            self._status_label.setStyleSheet("font-size: 11px; color: #27AE60;")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出{label}时出错:\n{str(e)}")

    def _export_as_mp3(self, source_path: str, save_path: str, label: str):
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "导出 MP3 需要 pydub 库和 ffmpeg。\n"
                "请运行: pip install pydub\n"
                "并确保 ffmpeg 在系统 PATH 中。"
            )
        audio = AudioSegment.from_wav(source_path)
        audio.export(save_path, format="mp3", bitrate="320k")

    # ═══════════════════════════════════════════════════════════════════════
    #  工具方法
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        total = int(seconds)
        h, m, s = total // 3600, (total % 3600) // 60, total % 60
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def _cleanup_temp_dir(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None
        if self._video_extracted_wav and os.path.isfile(self._video_extracted_wav):
            try:
                os.remove(self._video_extracted_wav)
            except Exception:
                pass
            self._video_extracted_wav = None

    def closeEvent(self, event):
        self._original_player.stop()
        self._separated_player.stop()
        self._cleanup_temp_dir()
        event.accept()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          程序入口                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("人声伴奏分离工具")
    font = QFont("Microsoft YaHei", 10)
    font.setFamilies(["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                       "WenQuanYi Micro Hei", "SimHei", "sans-serif"])
    app.setFont(font)
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
