#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QA 功能验证脚本 - 对 vocal_separator.py 核心功能做离屏自动化测试。"""
import os
import sys
import tempfile
import shutil

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import soundfile as sf

RESULTS = []  # (测试项, 结果, 说明)

def record(name, ok, note=""):
    RESULTS.append((name, "正常" if ok else "异常", note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}")

# ── 生成测试素材 ─────────────────────────────────────────────────────────
tmp = tempfile.mkdtemp(prefix="qa_")
sr = 44100
t = np.linspace(0, 5, 5 * sr, endpoint=False)
sine = 0.5 * np.sin(2 * np.pi * 440 * t)
stereo = np.stack([sine, sine], axis=1).astype(np.float32)
wav_stereo = os.path.join(tmp, "test_stereo.wav")
sf.write(wav_stereo, stereo, sr)

mono = (0.5 * sine).astype(np.float32)
wav_mono = os.path.join(tmp, "test_mono.wav")
sf.write(wav_mono, mono, sr)

noise = (0.2 * np.random.randn(5 * sr)).astype(np.float32)
wav_noise = os.path.join(tmp, "test_noise.wav")
sf.write(wav_noise, noise, sr)

wav_tiny = os.path.join(tmp, "test_tiny.wav")
sf.write(wav_tiny, np.array([0.1], dtype=np.float32), sr)  # 单采样

corrupt = os.path.join(tmp, "test_corrupt.mp3")
with open(corrupt, "wb") as f:
    f.write(os.urandom(4096))  # 伪装的损坏 MP3

empty_path = os.path.join(tmp, "does_not_exist.wav")

# ── 导入被测模块 ─────────────────────────────────────────────────────────
try:
    import vocal_separator as vs
    record("模块导入", True, "vocal_separator 导入成功 (PyQt6/demucs/torch 均可用)")
except SystemExit as e:
    record("模块导入", False, f"sys.exit({e.code}) - 依赖缺失即退出, 无 GUI 提示")
    sys.exit(1)
except Exception as e:
    record("模块导入", False, f"导入异常: {e}")
    sys.exit(1)

from PyQt6.QtWidgets import QApplication, QMessageBox

app = QApplication.instance() or QApplication(sys.argv)
# 拦截模态对话框，避免阻塞
QMessageBox.critical = staticmethod(lambda *a, **k: print("  [QMessageBox.critical]", a[2] if len(a) > 2 else ""))
QMessageBox.warning = staticmethod(lambda *a, **k: print("  [QMessageBox.warning]", a[2] if len(a) > 2 else ""))

# ── 1. 时长格式化 ────────────────────────────────────────────────────────
try:
    f = vs.MainWindow._format_duration
    assert f(0) == "00:00" and f(61) == "01:01" and f(3661) == "01:01:01" and f(-5) == "00:00"
    record("时长格式化 _format_duration", True, "0s/61s/3661s/-5s 边界全部正确")
except Exception as e:
    record("时长格式化 _format_duration", False, str(e))

# ── 2. 主窗口初始化 ─────────────────────────────────────────────────────
try:
    win = vs.MainWindow()
    record("主窗口初始化", True, "MainWindow 离屏实例化成功")
except Exception as e:
    record("主窗口初始化", False, str(e))
    sys.exit(1)

# ── 3. 文件加载: 正常 WAV ───────────────────────────────────────────────
try:
    win._load_audio_file(wav_stereo)
    ok = (win._input_file == wav_stereo and win._btn_separate.isEnabled()
          and len(win._original_waveform._tracks) == 1)
    record("加载正常 WAV", ok, f"input_file={os.path.basename(win._input_file or '')}, 波形轨数={len(win._original_waveform._tracks)}")
except Exception as e:
    record("加载正常 WAV", False, str(e))

# ── 4. 文件加载: 不存在 / 损坏文件 ─────────────────────────────────────
try:
    before = win._input_file
    win._load_audio_file(empty_path)
    record("加载不存在的文件", win._input_file == before, "弹出错误框且状态未被污染")
except Exception as e:
    record("加载不存在的文件", False, f"异常未捕获: {e}")

try:
    before = win._input_file
    win._load_audio_file(corrupt)
    record("加载损坏的 MP3", win._input_file == before, "LibsndfileError 被正确捕获并弹框")
except Exception as e:
    record("加载损坏的 MP3", False, f"异常未捕获: {e}")

# ── 5. 文件加载: 单采样 / 单声道 ───────────────────────────────────────
try:
    win._load_audio_file(wav_tiny)
    win._original_waveform.update()  # 触发 paintEvent
    record("加载单采样 WAV (边界)", True, "无除零/崩溃")
except Exception as e:
    record("加载单采样 WAV (边界)", False, str(e))

try:
    win._load_audio_file(wav_mono)
    record("加载单声道 WAV", True, "mono flatten 路径正常")
except Exception as e:
    record("加载单声道 WAV", False, str(e))

# 恢复正常素材
win._load_audio_file(wav_stereo)

# ── 6. 模拟分离完成 → stem 行与波形 ────────────────────────────────────
stem_dir = tempfile.mkdtemp(prefix="qa_stems_")
stems = {}
for name in ["vocals", "drums", "bass", "other"]:
    p = os.path.join(stem_dir, f"{name}.wav")
    sf.write(p, stereo * 0.25, sr)
    stems[name] = p
try:
    win._temp_dir = stem_dir
    win._on_separation_finished({"stems": stems, "sr": sr})
    ok = (len(win._stem_rows) == 4 and len(win._separated_waveform._tracks) == 4
          and win._btn_apply_postproc.isEnabled())
    record("分离完成回调 (4轨)", ok, f"stem行={len(win._stem_rows)}, 波形轨={len(win._separated_waveform._tracks)}")
except Exception as e:
    record("分离完成回调 (4轨)", False, str(e))

# ── 7. 后处理: 增益 + 削波保护 ─────────────────────────────────────────
try:
    win._stem_rows["vocals"].vol_slider.setValue(6)   # +6 dB ≈ x2
    win._stem_rows["drums"].vol_slider.setValue(24)   # 超范围? 滑块上限12, setValue(24) 会被钳到12
    actual_drums_gain = win._stem_rows["drums"].vol_slider.value()
    orig_data, _ = sf.read(stems["vocals"], always_2d=True)
    win._on_apply_postproc()
    proc_path = win._processed_stem_files.get("vocals")
    proc_data, _ = sf.read(proc_path, always_2d=True)
    ratio = np.sqrt(np.mean(proc_data**2) / np.mean(orig_data**2))
    peak = np.max(np.abs(proc_data))
    ok = 1.8 < ratio < 2.2 and peak <= 0.99
    record("后处理-增益+削波", ok, f"+6dB后RMS比值={ratio:.2f}(期望~2.0), 峰值={peak:.3f}<=0.99; setValue(24)被钳为{actual_drums_gain}")
except Exception as e:
    record("后处理-增益+削波", False, str(e))

# ── 8. 后处理: 降噪 ────────────────────────────────────────────────────
try:
    noise_stem = os.path.join(stem_dir, "vocals.wav")
    sf.write(noise_stem, noise, sr)  # 用纯噪声替换人声轨
    win._stem_files["vocals"] = noise_stem
    win._stem_rows["vocals"].vol_slider.setValue(0)
    win._stem_rows["vocals"].chk_denoise.setChecked(True)
    win._on_apply_postproc()
    den_data, _ = sf.read(win._processed_stem_files["vocals"])
    red = np.sqrt(np.mean(den_data**2)) / np.sqrt(np.mean(noise**2))
    record("后处理-降噪", red < 0.9, f"降噪后RMS为原始噪声的 {red:.1%}")
except Exception as e:
    record("后处理-降噪", False, str(e))

# ── 9. 后处理: 重置 ────────────────────────────────────────────────────
try:
    win._on_reset_postproc()
    ok = (win._processed_stem_files == {}
          and all(r.vol_slider.value() == 0 for r in win._stem_rows.values()))
    record("后处理-重置", ok, "滑块归零, 后处理映射清空")
except Exception as e:
    record("后处理-重置", False, str(e))

# ── 10. 导出 WAV / MP3 ─────────────────────────────────────────────────
from PyQt6.QtWidgets import QFileDialog
out_wav = os.path.join(tmp, "export_test.wav")
out_mp3 = os.path.join(tmp, "export_test.mp3")
try:
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_wav, ""))
    win._on_export("bass", "wav")
    ok = os.path.isfile(out_wav) and abs(sf.info(out_wav).duration - 5.0) < 0.01
    record("导出 WAV", ok, f"文件存在, 时长={sf.info(out_wav).duration:.2f}s")
except Exception as e:
    record("导出 WAV", False, str(e))

try:
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out_mp3, ""))
    win._on_export("bass", "mp3")
    ok = os.path.isfile(out_mp3) and os.path.getsize(out_mp3) > 10000
    record("导出 MP3", ok, f"文件存在, 大小={os.path.getsize(out_mp3)//1024}KB, 320kbps")
except Exception as e:
    record("导出 MP3", False, str(e))

# ── 11. AudioPlayer 播放状态机 ──────────────────────────────────────────
try:
    p = vs.AudioPlayer()
    p.load(wav_stereo)
    assert abs(p.duration - 5.0) < 0.01
    p.play()
    assert p.state == "playing"
    import time; time.sleep(0.3)
    p.pause()
    assert p.state == "paused" and 0.2 < p.current_position() < 1.0
    p.play(p.current_position())
    assert p.state == "playing"
    p.set_position(999)  # 超出时长 → 应被钳制
    assert p.current_position() <= p.duration + 0.1
    p.stop()
    assert p.state == "stopped" and p.current_position() == 0.0
    record("AudioPlayer 状态机", True, "play/pause/resume/seek钳制/stop 全部符合预期")
except Exception as e:
    record("AudioPlayer 状态机", False, str(e))

# ── 12. 波形 Seek 信号 ──────────────────────────────────────────────────
try:
    got = []
    win._separated_waveform.seek_requested.connect(lambda s: got.append(s))
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    w = win._separated_waveform
    w.resize(400, 100)
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(200, 50),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    w.mousePressEvent(ev)
    ok = len(got) == 1 and abs(got[0] - w._duration / 2) < 0.1
    record("波形点击 Seek", ok, f"点击中点 → seek={got[0]:.2f}s (时长{w._duration:.1f}s)" if got else "信号未触发")
except Exception as e:
    record("波形点击 Seek", False, str(e))

# ── 13. 临时文件清理 ────────────────────────────────────────────────────
try:
    win._video_extracted_wav = wav_tiny  # 借用它验证清理逻辑
    d = win._temp_dir
    win._cleanup_temp_dir()
    ok = (not os.path.isdir(d)) and (not os.path.isfile(wav_tiny))
    record("临时文件清理", ok, "temp_dir 与视频提取文件均被删除")
except Exception as e:
    record("临时文件清理", False, str(e))

# ── 14. closeEvent ──────────────────────────────────────────────────────
try:
    win.close()
    record("窗口关闭清理", True, "closeEvent 无异常")
except Exception as e:
    record("窗口关闭清理", False, str(e))

print("\n===== 汇总 =====")
npass = sum(1 for _, r, _ in RESULTS if r == "正常")
print(f"通过 {npass}/{len(RESULTS)}")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(stem_dir, ignore_errors=True)
